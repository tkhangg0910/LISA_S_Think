# What is in this archive, and how to put it on GitHub

```
README.md            the main document — the task, how the reasoning chain works, results
REPRODUCE.md         exact commands for every column of the results table
PUSH_TO_GITHUB.md    this file
results/             matrix.csv + the metrics.json behind every published number
docs/SETUP_vi.md     HiPerGator environment notes (Vietnamese)

lisa/                → becomes branch  main       (LISA-7B, LISA-13B)
lisa-plus/           → becomes branch  lisa_plus  (LISA++-7B)
```

`lisa/` and `lisa-plus/` are each a complete, runnable tree. They are separate
because LISA++ ships a different multimodal projector (`mlp2x_gelu`) and a 336px
CLIP tower, and only the `lisa_plus` code path rebuilds them — the `main` tree
loads a LISA++ checkpoint without complaint and randomises the projector.
Each tree already contains its own copy of `README.md`, `REPRODUCE.md`,
`results/` and `docs/`, so whichever branch a reader lands on is self-contained.

Not included, by design: `checkpoints/` (model weights), `data/` (the 386 MB
parquet split — one file is 260 MB, over GitHub's 100 MB per-file limit), `runs/`,
`logs/`, `env*/`. All of them are fetched by the scripts in `setup/`.

---

## Option A — fork upstream (recommended)

Keeps the upstream history and attribution, and gives you the `lisa_plus` branch
for free, since it is upstream's own branch.

```bash
# 1. Fork https://github.com/dvlab-research/LISA on GitHub, then:
git clone git@github.com:<you>/LISA.git lisa-brain && cd lisa-brain

# 2. main branch
git checkout main
cp -r /path/to/LISA-brain-mri/lisa/. .
git add -A
git commit -m "Brain-MRI atrophy segmentation: reasoning-guided fine-tune of LISA 7B/13B"
git push origin main

# 3. lisa_plus branch
git checkout lisa_plus
cp -r /path/to/LISA-brain-mri/lisa-plus/. .
git add -A
git commit -m "Brain-MRI atrophy segmentation: reasoning-guided fine-tune of LISA++ 7B"
git push origin lisa_plus
```

Check before pushing that no weights or parquet slipped in:

```bash
git ls-files | xargs -r du -ch 2>/dev/null | tail -1     # should be a few MB
git ls-files | grep -E '\.(bin|pth|safetensors|parquet)$'  # should print nothing
```

## Option B — standalone repo, no upstream history

```bash
cd /path/to/LISA-brain-mri/lisa
git init -b main
git add -A && git commit -m "LISA brain-MRI: reasoning-guided atrophy segmentation"
git remote add origin git@github.com:<you>/<repo>.git
git push -u origin main

git checkout --orphan lisa_plus
git rm -rf --cached . >/dev/null && rm -rf ./* .gitignore
cp -r ../lisa-plus/. .
git add -A && git commit -m "LISA++ brain-MRI: reasoning-guided atrophy segmentation"
git push -u origin lisa_plus
git checkout main
```

The `rm -rf ./*` in the middle deletes the working tree on purpose — an orphan
branch starts empty. Run it only inside the throwaway `lisa/` copy from this
archive, never in a directory you care about.

---

## Changes made when packaging

The trees here are not byte-identical to the ones the results were produced on.
Three mechanical changes were applied so the code runs outside one specific
HiPerGator account, and nothing else was touched — no model code, no dataset code,
no loss, no hyper-parameter *values*:

1. **Absolute paths removed.** 31 files hard-coded
   `/blue/rc-rse/.../LISA`. Shell scripts now derive `LISA_ROOT` from their own
   location; SLURM job scripts take `SLURM_SUBMIT_DIR` (so submit from the repo
   root). `#SBATCH --output=` was made relative, since SLURM does not expand
   variables in `#SBATCH` directives.
2. **Published hyper-parameters baked into the job scripts.** The original
   `slurm/train_brain_b200.sbatch` carried older defaults (`lr 1e-4`, 20 epochs,
   `model_max_length 512`, frozen SAM encoder, no `--use_q1`) and the real runs
   overrode all of them on the `sbatch` command line. Anyone running the old file
   as-is would have trained a different, weaker model and had no way to know. The
   defaults now are the published configuration. `slurm/train_brain_nr.sbatch` was
   added for the non-reasoning ablation.
3. **`data/` added to `.gitignore`**, and `worktrees/` removed from it — the
   LISA++ tree is a branch now, not a checked-out worktree.

The three LISA++ job scripts moved from `main`'s `slurm/` to the `lisa_plus`
branch and no longer `cd` into a worktree directory.
