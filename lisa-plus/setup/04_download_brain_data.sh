#!/bin/bash
# Download the tkhangg0910/brain_data_split_v1 HF dataset into $LISA_ROOT/data.
# curl, not huggingface_hub -- see the header of 02_download_weights.sh for why.
set -euo pipefail
LISA_ROOT="${LISA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

REPO=tkhangg0910/brain_data_split_v1
HF="https://huggingface.co/datasets/$REPO/resolve/main"
DST="$LISA_ROOT/data/brain_data_split_v1"
mkdir -p "$DST/data"

FILES=$(curl -sS -m 60 "https://huggingface.co/api/datasets/$REPO" \
    | python3 -c 'import json,sys; [print(s["rfilename"]) for s in json.load(sys.stdin)["siblings"]]')

for f in $FILES; do
    [ "$f" = ".gitattributes" ] && continue
    out="$DST/$f"
    mkdir -p "$(dirname "$out")"
    remote=$(curl -sSLI -m 60 "$HF/$f" | tr -d '\r' \
             | awk 'tolower($1)=="content-length:"{n=$2} END{print n+0}')
    local_sz=$(stat -c %s "$out" 2>/dev/null || echo 0)
    if [ "$remote" -gt 0 ] && [ "$local_sz" -eq "$remote" ]; then
        printf '    %-45s skip (%s bytes)\n' "$f" "$remote"
        continue
    fi
    printf '    %-45s downloading %s bytes\n' "$f" "$remote"
    curl -sSL --retry 5 --retry-all-errors -C - -o "$out" "$HF/$f"
    got=$(stat -c %s "$out")
    [ "$got" -eq "$remote" ] || { echo "SIZE MISMATCH $f: $got != $remote" >&2; exit 1; }
done

echo; echo "==> $DST"; find "$DST" -type f -exec ls -la {} \; ; du -sh "$DST"
