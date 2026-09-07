# Reasoning-guided atrophy segmentation on brain MRI with LISA

Fine-tunes LISA-7B, LISA-13B and LISA++-7B to answer a clinical question about an
axial brain-MRI slice **and** segment three atrophy markers in the same forward
pass. The point of the project is the middle term: the model is trained to emit a
diagnostic reasoning chain *before* the segmentation tokens, and that chain is
what conditions the mask decoder.

| Marker | Full name | Non-empty in test split |
|---|---|---|
| MTA | medial temporal atrophy | 31 / 542 |
| GCA | global cortical atrophy | 307 / 542 |
| Koedam | posterior atrophy | 130 / 542 |

Data: [`tkhangg0910/brain_data_split_v1`](https://huggingface.co/datasets/tkhangg0910/brain_data_split_v1),
1000 train / 542 test rows, used exactly as shipped.

---

## 1. How the "thinking" works

### 1.1 What the dataset gives us

Each slice ships a four-turn annotation chain:

| Turn | Content |
|---|---|
| Q1 / **A1** | *localisation* — normalised `[x1,y1,x2,y2]` boxes over the regions of interest |
| Q2 / **A2** | *findings* — free-text radiological description |
| Q3 / **A3** | *severity* — visual-rating-scale scores, e.g. `GCA = 1, Koedam = 2` |
| Q4 / **A4** | *conclusion* — the diagnostic status (`Non-Dementia` … `Moderate-Dementia`) |

plus three binary masks (MTA, GCA, Koedam), any of which may be empty.

### 1.2 What the model sees

**Input is Q4 and the image only.** A1–A3 never appear in the prompt. They are
distilled into the *target*, so the model has to produce the chain itself, and the
chain is the thing that carries information forward to the masks.

The target is assembled by `build_answer()` in `utils/brain_seg_dataset.py`:

```
Regions of interest: (x1,y1,x2,y2), …          ← A1   (only with --use_q1)
Findings: <A2>                                  ← A2
There is <present markers>; there is no <absent markers>.   ← from the MASKS
Severity: <A3>.                                 ← A3
Therefore the pathological impact is Status (<A4>).
Medial temporal atrophy: [SEG]. Global cortical atrophy: [SEG]. Posterior atrophy: [SEG].
```

A real training target, row `test_OAS1_0027_mpr-3_158`:

> **Q4:** How would you determine the pathological impact?
>
> **Target:** Regions of interest: (0.61,0.34,0.73,0.47), (0.5,0.53,0.75,0.62),
> (0.24,0.53,0.33,0.58), (0.25,0.6,0.46,0.74), (0.48,0.6,0.71,0.75). Findings:
> Substantial sulcal widening of posterior cingulate and parieto-occipital sulci,
> substantial gyral atrophy. There is global cortical atrophy, posterior atrophy;
> there is no medial temporal atrophy. Severity: Koedam = 2. Therefore the
> pathological impact is Status (Moderate-Dementia). Medial temporal atrophy:
> `[SEG]`. Global cortical atrophy: `[SEG]`. Posterior atrophy: `[SEG]`.

### 1.3 Why the chain must come *before* the `[SEG]`

Attention is causal. Every token emitted before a `[SEG]` is visible to the hidden
state that produces that `[SEG]`, and that hidden state is literally the prompt
handed to SAM:

```
image ─┬─→ SAM image_encoder (1024px) ─────────────────→ image_embeddings ──┐
       └─→ CLIP tower (224px) → mm_projector → Llama ──→ hidden states      │
                                                              │             │
                                     hidden state at each [SEG] token       │
                                                              ↓             │
                                     text_hidden_fcs: 4096 → 256   (LISA.py:311)
                                                              ↓             ↓
                        SAM prompt_encoder(text_embeds=…) → mask_decoder → mask
```

So the reasoning text is not commentary printed alongside the mask — it is the
conditioning signal *for* the mask. Putting it after the `[SEG]` (`--answer_order
seg_first`) or removing it (`--answer_order a4_seg`) changes what the mask decoder
can condition on. That is exactly what the ablations in §2 measure.

### 1.4 Three design decisions worth knowing

1. **Always exactly three `[SEG]`,** in the fixed order MTA → GCA → Koedam, with
   an all-zero ground-truth mask where a marker is absent. LISA asserts
   `n_[SEG] == n_masks`. Measured non-empty masks per row are
   `train {0:244, 1:387, 2:368, 3:1}`, `test {0:207, 1:202, 2:133, 3:0}` — a
   variable-`[SEG]` scheme would leave MTA (empty in 93% of rows) with no
   prediction on most samples and its per-class IoU undefined. Fixing the count at
   three makes every metric well defined everywhere.

2. **`dice_loss_weight` 0.5 → 2.0.** Lesions occupy 0.5–5.6% of the image (MTA
   median 0.53%, Koedam 1.27%, GCA 5.63%). At that imbalance BCE alone collapses
   to predicting empty.

3. **The presence sentence is derived from the masks, not from A3.** A3 disagrees
   with the mask annotations on 40–75% of rows depending on the marker (e.g. 142
   rows have `GCA = 0` yet a non-empty GCA mask). The masks are what gets scored,
   so they are the source of truth.

### 1.5 The switches

| Flag | Values | Effect on the chain |
|---|---|---|
| `--use_q1` | off / **on** | prepend A1's localisation boxes |
| `--answer_order` | **`reason_first`** / `seg_first` / `a4_seg` | chain before `[SEG]` / after / removed entirely |
| `--clip_view` | **`center_crop`** / `pad_square` / `tight_pad` | how the slice is framed for the LLM branch |
| `--neutral_reasoning` | eval only | strip the chain from the forced answer at test time |

Bold = the published configuration. `--answer_order` and `--use_q1` **must match
between training and evaluation** — `eval_brain_seg.py` takes the same flags and
does not read them back from the checkpoint.

---

## 2. Does the reasoning chain actually help?

`a4_seg` is the like-for-like ablation: same data, same recipe, same schedule, but
the target is only `Status (<label>).` + three `[SEG]` — no boxes, no findings, no
presence sentence, no severity. It was trained for **80 epochs against the
reasoning runs' 40**, so it is not handicapped by budget.

DICE on samples with non-empty ground truth:

| | MTA | GCA | Koedam | Overall |
|---|---|---|---|---|
| **7B** with chain | 0.8650 | **0.7455** | **0.7039** | **0.7419** |
| 7B without (`a4_seg`) | 0.8655 | 0.6992 | 0.5046 | 0.6562 |
| **13B** with chain | 0.8592 | **0.7659** | **0.7251** | **0.7607** |
| 13B without (`a4_seg`) | 0.8674 | 0.7053 | 0.4201 | 0.6368 |

The gain is not uniform, and the shape of it is the interesting result:

- **MTA: no effect at all** (7B −0.0005, 13B −0.0082 — the ablation is marginally
  *better*). MTA is the marker with 31 positive samples; there is nothing for the
  chain to add.
- **GCA: modest**, +0.046 / +0.061.
- **Koedam: large**, +0.199 / +0.305. Koedam is also the marker the LLM branch is
  worst at seeing — with the stock `center_crop` view, 40% of Koedam lesion pixels
  fall outside the 224×224 centre crop (see `make_square()` in
  `utils/brain_seg_dataset.py`). The chain appears to be compensating for a visual
  pathway that cannot see the lesion.

**Read that last point carefully before quoting the numbers.** See §4.

---

## 3. Results

`results/matrix.csv`, columns `LISA (7B/13B) (-R)` and `LISA++ (7B/13B) (-R)`,
regenerated from `results/*.metrics.json` by `scripts/build_matrix.py` — never
typed by hand.

| Metric | | LISA 7B/13B | LISA++ 7B | LISA 7B/13B, no chain |
|---|---|---|---|---|
| Overall | Dice | 0.7419 / 0.7607 | 0.7373 | 0.6562 / 0.6368 |
| Overall | IoU | 0.6171 / 0.6352 | 0.6108 | 0.5415 / 0.5225 |
| MTA | Dice | 0.8650 / 0.8592 | 0.8670 | 0.8655 / 0.8674 |
| MTA | IoU | 0.7658 / 0.7561 | 0.7687 | 0.7655 / 0.7693 |
| GCA | Dice | 0.7455 / 0.7659 | 0.7395 | 0.6992 / 0.7053 |
| GCA | IoU | 0.6264 / 0.6462 | 0.6187 | 0.5821 / 0.5842 |
| Koedam | Dice | 0.7039 / 0.7251 | 0.7011 | 0.5046 / 0.4201 |
| Koedam | IoU | 0.5599 / 0.5803 | 0.5545 | 0.3921 / 0.3179 |

`IoU` is mean per-sample IoU (LISA's code calls it gIoU). Population = samples with
non-empty ground truth. There is no LISA++ 13B column because no LISA++ 13B
checkpoint has been released.

---

## 4. Limitations — please read before citing

1. **Evaluation is teacher-forced.** The answer, including its reasoning chain and
   its three `[SEG]`, is fed to the model; the model only produces the masks. This
   keeps the *k*-th `[SEG]` bound to the *k*-th marker so per-class numbers are
   comparable across checkpoints, but it means the chain is *supplied* at test
   time rather than generated. Stripping it at eval (`--neutral_reasoning`, LISA++)
   drops overall Dice 0.7373 → 0.5517 and Koedam 0.7011 → **0.0354**. A deployment
   that must generate its own chain is not measured by this table.
2. **The split leaks.** All three test patients also appear in train (9 patients
   total), and 58 of 174 test slices appear in train differing only by an `mpr-N`
   rotation. These are internal comparisons between configurations, not
   generalisation estimates.
3. **Checkpoints are selected on the test set.** `train_ds.py` saves whenever
   validation gIoU improves, and validation here *is* the test split — there is no
   third split without re-partitioning.
4. **`utils/dataset.py` hard-codes a 255-token image expansion** in `collate_fn`'s
   truncation. Correct for the 224px CLIP tower used on `main`; the `lisa_plus`
   branch uses 575 for its 336px tower. If you ever run a 336px tower on `main`,
   the trailing `[SEG]` gets truncated on a small fraction of rows and training
   dies at a random step with a `gt_mask.shape` / `pred_mask.shape` assertion.

---

## 5. Repository layout

The three columns do **not** come from one code tree. LISA++ ships a different
projector (`mlp2x_gelu`) and a 336px CLIP tower, and only the upstream
`lisa_plus` branch rebuilds them correctly — the `main` tree would randomise the
projector silently, producing plausible-looking but wrong results.

```
branch  main        LISA-7B, LISA-13B
branch  lisa_plus   LISA++-7B   (forked from upstream's own lisa_plus branch)
```

Both branches carry the same auxiliary files:

```
results/       matrix.csv + the metrics.json behind every number above
docs/          SETUP_vi.md — HiPerGator environment notes (Vietnamese)
REPRODUCE.md   exact commands, per column
```

What was added on top of upstream LISA, in both trees:

| File | Role |
|---|---|
| `utils/brain_seg_dataset.py` | the dataset — prompt/target construction, `build_answer()` |
| `scripts/eval_brain_seg.py` | per-class IoU/DICE → `metrics.json` |
| `scripts/restore_lisa_modules.py` | **required**; see below |
| `scripts/export_trained_model.sh` | ZeRO shards → fp32 → merge LoRA → HF checkpoint |
| `scripts/build_matrix.py` | `metrics.json` → `matrix.csv` |
| `slurm/*.sbatch` | job scripts, published hyper-parameters baked in |
| `setup/*.sh`, `env.sh` | environment and weight/data download |

and four upstream files are patched: `model/LISA.py`, `train_ds.py`,
`utils/dataset.py`, plus `merge_lora_weights_and_save_hf_model.py` on `lisa_plus`.

### Two upstream bugs that block this use case

Both bite only when fine-tuning *from a LISA checkpoint* rather than from plain
LLaVA, and neither announces itself:

- `initialize_lisa_modules()` **constructs** a fresh SAM and a randomly-initialised
  `text_hidden_fcs`, overwriting what `from_pretrained` just loaded — the trained
  segmentation head silently restarts from scratch. Loss just starts high.
  `--restore_lisa_modules` copies those weights back.
- `ce/dice/bce_loss_weight` were assigned only in the `if not hasattr(config,
  "train_mask_decoder")` branch, i.e. only on a LLaVA start. Fine-tuning from LISA
  took the `else` branch and crashed on the first backward with
  `AttributeError: 'LISAForCausalLM' object has no attribute 'ce_loss_weight'`.

---

## 6. Quick start

```bash
git clone <this repo> && cd <repo>
bash setup/01b_create_env_b200.sh          # conda env, everything stays in-tree
bash setup/02_download_weights.sh          # LISA-7B, SAM ViT-H, CLIP 224
bash setup/05_download_more_bases.sh       # LISA-13B, LISA++ 7B, CLIP 336
bash setup/04_download_brain_data.sh       # the parquet split (386 MB, not in git)
sbatch slurm/train_brain_b200.sbatch --exp_name brainM-q1
```

Full per-column commands: **[REPRODUCE.md](REPRODUCE.md)**.

Hardware: every run reported here used a single NVIDIA B200 (180 GB), bf16,
DeepSpeed ZeRO. To fit a smaller card, scale `--batch_size` and
`--grad_accumulation_steps` inversely — keep their product at 20 so one epoch
remains exactly one pass over the 1000 training rows.

## Licence and attribution

Upstream [LISA](https://github.com/dvlab-research/LISA) (Apache 2.0); `lisa_plus`
is upstream's own branch. The brain-MRI dataset belongs to its authors and is
downloaded at setup time, not redistributed here.
