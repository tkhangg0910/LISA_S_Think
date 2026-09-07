#!/bin/bash
# Extra base checkpoints for the matrix: LISA-13B and LISA++ 7B, plus the 336px CLIP
# tower that LISA++ was trained against. curl, not huggingface_hub -- see 02_*.sh.
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$LISA_ROOT/env.sh"
CKPT="$LISA_ROOT/checkpoints"; HF="https://huggingface.co"

fetch(){  # fetch <repo> <dst> [regex-filter]
  local repo="$1" dst="$2" filt="${3:-.}"
  mkdir -p "$dst"; echo -e "\n==> $repo -> $dst"
  curl -sS -m 60 "$HF/api/models/$repo" \
    | python3 -c 'import json,sys;[print(s["rfilename"]) for s in json.load(sys.stdin)["siblings"]]' \
    | grep -E "$filt" | grep -vE '\.(msgpack|h5)$' | while read -r f; do
      out="$dst/$f"; mkdir -p "$(dirname "$out")"
      remote=$(curl -sSLI -m 60 "$HF/$repo/resolve/main/$f" | tr -d '\r' \
               | awk 'tolower($1)=="content-length:"{n=$2} END{print n+0}')
      local_sz=$(stat -c %s "$out" 2>/dev/null || echo 0)
      if [ "$remote" -gt 0 ] && [ "$local_sz" -eq "$remote" ]; then
        printf '    %-42s skip\n' "$f"; continue; fi
      printf '    %-42s %s bytes\n' "$f" "$remote"
      curl -sSL --retry 5 --retry-all-errors -C - -o "$out" "$HF/$repo/resolve/main/$f"
      got=$(stat -c %s "$out"); [ "$got" -eq "$remote" ] || { echo "SIZE MISMATCH $f" >&2; exit 1; }
    done
}

fetch xinlai/LISA-13B-llama2-v1        "$CKPT/LISA-13B-llama2-v1"
fetch Senqiao/LISA_Plus_7b             "$CKPT/LISA_Plus_7b"
# LISA++ was trained against the 336px tower, not the 224px one the other models use
fetch openai/clip-vit-large-patch14-336 "$CKPT/clip-vit-large-patch14-336" \
      '\.(json|txt|model)$|^pytorch_model\.bin$'

# point each config at the local CLIP copy so nothing resolves over the network at runtime
python3 - <<'PY'
import json, os
ckpt = os.path.join(os.environ["LISA_ROOT"], "checkpoints")
for name, clip in [("LISA-13B-llama2-v1", "clip-vit-large-patch14"),
                   ("LISA_Plus_7b",       "clip-vit-large-patch14-336")]:
    p = os.path.join(ckpt, name, "config.json")
    if not os.path.exists(p): continue
    cfg = json.load(open(p)); local = os.path.join(ckpt, clip); ch = False
    for k in ("vision_tower", "mm_vision_tower"):
        if k in cfg and cfg[k] != local:
            print(f"  {name}: {k}: {cfg[k]} -> {local}"); cfg[k] = local; ch = True
    if ch: json.dump(cfg, open(p, "w"), indent=2)
PY
echo; du -sh "$CKPT"/* 2>/dev/null || true
