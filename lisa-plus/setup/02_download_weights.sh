#!/bin/bash
# Download every weight LISA-7B needs into $LISA_ROOT/checkpoints.
# Safe to re-run: curl -C - resumes, complete files are skipped.
#
# WHY CURL AND NOT huggingface_hub:
#   On this cluster's compute nodes, curl reaches huggingface.co in ~0.3 s, but
#   huggingface_hub's python HTTP stack takes ~80 s per connection and
#   snapshot_download() wedges outright (0 bytes after 4 minutes, repeatedly).
#   curl against the plain /resolve/ URLs is fast, resumable and observable.
set -euo pipefail

LISA_ROOT="${LISA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$LISA_ROOT/env.sh"

CKPT="$LISA_ROOT/checkpoints"
mkdir -p "$CKPT"

HF=https://huggingface.co

# list_files <repo> -> one filename per line (needs no python networking)
list_files() {
    curl -sS -m 60 "$HF/api/models/$1" \
        | python -c 'import json,sys; [print(s["rfilename"]) for s in json.load(sys.stdin)["siblings"]]'
}

# fetch_repo <repo> <dest-dir> [grep-filter]
fetch_repo() {
    local repo="$1" dst="$2" filter="${3:-.}"
    mkdir -p "$dst"
    echo -e "\n==> $repo  ->  $dst"
    local files
    files=$(list_files "$repo" | grep -E "$filter" | grep -vE '\.(msgpack|h5)$')
    while read -r f; do
        [ -z "$f" ] && continue
        local out="$dst/$f"
        mkdir -p "$(dirname "$out")"
        # skip if we already have it at full remote size
        local remote local_sz
        remote=$(curl -sSLI -m 60 "$HF/$repo/resolve/main/$f" \
                 | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n+0}')
        local_sz=$(stat -c %s "$out" 2>/dev/null || echo 0)
        if [ "$remote" -gt 0 ] && [ "$local_sz" -eq "$remote" ]; then
            printf '    %-45s skip (%s bytes)\n' "$f" "$remote"
            continue
        fi
        printf '    %-45s downloading %s bytes\n' "$f" "$remote"
        curl -sSL --retry 5 --retry-all-errors -C - -o "$out" "$HF/$repo/resolve/main/$f"
    done <<< "$files"
}

# 1) LISA-7B reasoning-segmentation checkpoints (merged, HF format) -----------
fetch_repo xinlai/LISA-7B-v1-explanatory "$CKPT/LISA-7B-v1-explanatory"
fetch_repo xinlai/LISA-7B-v1             "$CKPT/LISA-7B-v1"

# 2) CLIP vision tower --------------------------------------------------------
# Kept as a plain local directory instead of an HF cache entry, so nothing has
# to resolve "openai/clip-vit-large-patch14" over the network at run time.
fetch_repo openai/clip-vit-large-patch14 "$CKPT/clip-vit-large-patch14" \
    '\.(json|txt|model)$|^pytorch_model\.bin$'

# 3) Point the LISA configs at that local CLIP directory ----------------------
python - <<'PY'
import json, os
ckpt = os.path.join(os.environ["LISA_ROOT"], "checkpoints")
clip = os.path.join(ckpt, "clip-vit-large-patch14")
for name in ("LISA-7B-v1-explanatory", "LISA-7B-v1"):
    p = os.path.join(ckpt, name, "config.json")
    if not os.path.exists(p):
        continue
    cfg = json.load(open(p))
    changed = False
    for key in ("vision_tower", "mm_vision_tower"):
        if cfg.get(key) not in (None, clip):
            print(f"  {name}: {key}: {cfg[key]} -> {clip}")
            cfg[key] = clip
            changed = True
    if changed:
        json.dump(cfg, open(p, "w"), indent=2)
print("==> configs point at the local CLIP copy")
PY

# 4) SAM ViT-H (training only; the inference checkpoints already carry SAM) ----
SAM="$CKPT/sam_vit_h_4b8939.pth"
if [ ! -s "$SAM" ]; then
    echo -e "\n==> SAM ViT-H -> $SAM"
    curl -sSL --retry 5 --retry-all-errors -C - -o "$SAM" \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
else
    echo -e "\n==> SAM ViT-H already present, skipping"
fi

echo
echo "==> checkpoints/"
du -sh "$CKPT"/* 2>/dev/null || true
