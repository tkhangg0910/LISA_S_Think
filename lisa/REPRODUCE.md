# Reproducing every column of `results/matrix.csv`

All commands are submitted **from the repository root** — the SLURM scripts take
`SLURM_SUBMIT_DIR` as `LISA_ROOT`. Outside SLURM, `export LISA_ROOT=$PWD` first.

The hyper-parameters below are baked into the `.sbatch` files; the only thing you
pass is the experiment name (and, for 13B, the base checkpoint). This is a change
from how the runs were originally launched, where the values were appended on the
command line — the defaults in the job scripts now *are* the published
configuration, so there is nothing to remember.

---

## 0. Setup, once

```bash
bash setup/01b_create_env_b200.sh     # conda env inside the repo (torch 2.7.1+cu128)
bash setup/02_download_weights.sh     # LISA-7B-v1, SAM ViT-H, CLIP ViT-L/14 (224)
bash setup/05_download_more_bases.sh  # LISA-13B-llama2-v1, LISA_Plus_7b, CLIP 336
bash setup/04_download_brain_data.sh  # brain_data_split_v1 parquet (386 MB)
```

The parquet files are not in git (260 MB exceeds GitHub's 100 MB per-file limit);
`04_download_brain_data.sh` pulls them from the Hugging Face Hub and verifies
`content-length` on every file.

---

## 1. `LISA (7B) (-R)` — overall Dice 0.7419, IoU 0.6171

Branch `main`.

```bash
# train  (~40 epochs; batch 2 x grad_accum 10 x 50 steps = one pass/epoch)
sbatch slurm/train_brain_b200.sbatch --exp_name brainM-q1

# export: ZeRO shards -> fp32 -> merge LoRA -> runs/brainM-q1/hf_model
sbatch slurm/export_brain.sbatch brainM-q1

# evaluate
sbatch slurm/eval_brain_b200.sbatch \
    --version runs/brainM-q1/hf_model \
    --use_q1 --answer_order reason_first --clip_view center_crop \
    --out runs/brainM-q1/metrics.json
```

## 2. `LISA (13B) (-R)` — overall Dice 0.7607, IoU 0.6352

Identical recipe, different base checkpoint.

```bash
LISA_BASE=$PWD/checkpoints/LISA-13B-llama2-v1 \
    sbatch slurm/train_brain_b200.sbatch --exp_name brain13B-q1

sbatch slurm/export_brain13b.sbatch          # passes the 13B base to the merge step

sbatch slurm/eval_brain_b200.sbatch \
    --version runs/brain13B-q1/hf_model \
    --use_q1 --answer_order reason_first --clip_view center_crop \
    --out runs/brain13B-q1/metrics.json
```

## 3. `LISA++ (7B) (-R)` — overall Dice 0.7373, IoU 0.6108

**Branch `lisa_plus`.** This cannot be run from `main`: only this branch's
`model/llava1p5` and its `merge_lora_weights_and_save_hf_model.py` rebuild the
`mlp2x_gelu` projector LISA++ was trained with. Running it from `main` produces a
model that loads, trains and scores — with a randomised projector.

```bash
git checkout lisa_plus

sbatch slurm/train_brain_plus.sbatch --exp_name brainPlus7B-q1
sbatch slurm/export_brain_plus.sbatch brainPlus7B-q1

sbatch slurm/eval_brain_plus.sbatch \
    --version runs/brainPlus7B-q1/hf_model \
    --vision-tower $PWD/checkpoints/clip-vit-large-patch14-336 \
    --no-use_mm_start_end \
    --use_q1 --answer_order reason_first --clip_view center_crop \
    --out runs/brainPlus7B-q1/metrics.json
```

Three differences from `main`, all forced by the checkpoint:

| | `main` | `lisa_plus` |
|---|---|---|
| CLIP tower | `clip-vit-large-patch14` (224px, 256 patches) | `clip-vit-large-patch14-336` (576 patches) |
| `<im_start>`/`<im_end>` | present, vocab 32003–32004 | **absent**, vocab 32001 → `--no-use_mm_start_end` everywhere |
| `model_max_length` | 1024 | **1536** — `collate_fn` budgets `len - 575`; at 1024 the trailing `[SEG]` is truncated on ~0.2% of rows and training dies at a random step |

## 4. Ablation: `LISA (7B/13B) ours-NR` — the chain removed

Branch `main`. Same recipe, `--answer_order a4_seg`, no `--use_q1`, and **80
epochs instead of 40** so the ablation is not budget-limited.

```bash
sbatch slurm/train_brain_nr.sbatch --exp_name brainNR7B-a4seg
LISA_BASE=$PWD/checkpoints/LISA-13B-llama2-v1 \
    sbatch slurm/train_brain_nr.sbatch --exp_name brainNR13B-a4seg

sbatch slurm/export_brain.sbatch brainNR7B-a4seg
# 13B: bash scripts/export_trained_model.sh brainNR13B-a4seg $PWD/checkpoints/LISA-13B-llama2-v1

sbatch slurm/eval_brain_b200.sbatch --version runs/brainNR7B-a4seg/hf_model \
    --answer_order a4_seg --clip_view center_crop \
    --out runs/brainNR7B-a4seg/metrics.json
```

Note the eval flags mirror training: no `--use_q1`, `--answer_order a4_seg`.
`eval_brain_seg.py` does not read these back from the checkpoint, and a mismatch
gives wrong numbers without any error.

## 5. Ablation: how much of the score is the text?

Re-evaluate a *trained* model with the reasoning stripped from the forced answer:

```bash
sbatch slurm/eval_brain_plus.sbatch \
    --version runs/brainPlus7B-q1/hf_model \
    --vision-tower $PWD/checkpoints/clip-vit-large-patch14-336 \
    --no-use_mm_start_end --use_q1 --answer_order reason_first \
    --neutral_reasoning \
    --out runs/brainPlus7B-q1/metrics_neutral.json
```

Result (`results/brainPlus7B-q1.metrics_neutral.json`): overall Dice
0.7373 → 0.5517, Koedam 0.7011 → 0.0354. See README §4.

## 6. Rebuild the table

```bash
python scripts/build_matrix.py     # runs/*/metrics.json -> matrix.csv
```

The `(-R)` and `ours-NR` columns are read out of `metrics.json`; the published
baselines in the other columns are transcribed constants in the `BASE` dict at the
top of that script.

---

## Verifying without retraining

Every number in the README is in `results/*.metrics.json`, so the table can be
checked without a GPU:

```bash
python - <<'PY'
import json, glob, os
for p in sorted(glob.glob("results/*.metrics.json")):
    m = json.load(open(p))
    print(os.path.basename(p))
    print("  meta:", m["_meta"])
    for k in ("OVERALL", "MTA", "GCA", "Koedam"):
        r = m[k]["present"]
        print(f"  {k:<8} n={r['n']:<4} gIoU={r['gIoU']:.4f}  DICE={r['DICE']:.4f}")
PY
```

Each file also carries `per_sample` (one record per non-empty-GT sample+class,
with its IoU, Dice, GT pixel count and predicted pixel count) and IoU percentiles,
so the distribution behind each mean can be inspected rather than taken on trust.

---

## Recovering the original launch commands

The values baked into the job scripts were taken from the DeepSpeed launcher line
in the SLURM logs of the original runs, not retyped from memory. For the record,
`brainM-q1` was launched as the job-script defaults of the time plus these
overrides:

```
--lr 3e-4 --epochs 40 --train_vision_encoder_blocks 8 --batch_size 2 \
--grad_accumulation_steps 10 --steps_per_epoch 50 --model_max_length 1024 \
--use_q1 --answer_order reason_first --clip_view center_crop
```

One caveat, stated for honesty: SLURM does not record the arguments passed to
`sbatch`, so the **evaluation** command lines could not be recovered verbatim.
`--use_q1`, `--answer_order` and `--clip_view` are read back from the `_meta`
block that `eval_brain_seg.py` writes into each `metrics.json`, so those are
certain. The two LISA++-only flags (`--vision-tower`, `--no-use_mm_start_end`) are
inferred from the code paths and from `vocab_size: 32001` in the exported
checkpoint's `config.json` — necessary for the run to have worked at all, but not
directly attested in a log.
