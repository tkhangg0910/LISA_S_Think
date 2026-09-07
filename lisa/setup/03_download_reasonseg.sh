#!/bin/bash
# Download the ReasonSeg dataset (train 239 / val 200) into $LISA_ROOT/dataset.
#
# ReasonSeg is hosted on Google Drive, which rate-limits aggressively. If gdown
# fails, download the folder manually from
#   https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy
# and unzip it so the layout is:
#   dataset/reason_seg/ReasonSeg/{train,val,explanatory}
set -euo pipefail

LISA_ROOT="${LISA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$LISA_ROOT/env.sh"

DST="$LISA_ROOT/dataset/reason_seg"
mkdir -p "$DST"
cd "$DST"

FOLDER_ID="125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy"

echo "==> gdown folder $FOLDER_ID -> $DST"
gdown --folder "https://drive.google.com/drive/folders/$FOLDER_ID" -O "$DST" --remaining-ok || {
    echo "gdown failed -- download manually, see the header of this script." >&2
    exit 1
}

# unzip whatever came down
shopt -s nullglob
for z in "$DST"/*.zip "$DST"/*/*.zip; do
    echo "==> unzip $z"
    unzip -n -q "$z" -d "$DST"
done

# The zips extract to train/ val/ test/ explanatory/ at the top level, but the
# code wants one more level: reason_seg/ReasonSeg/{train,val,explanatory}
# (utils/reason_seg_dataset.py joins base_image_dir + "reason_seg" + "ReasonSeg").
mkdir -p "$DST/ReasonSeg"
for d in train val test explanatory; do
    [ -d "$DST/$d" ] && mv "$DST/$d" "$DST/ReasonSeg/"
done
rm -f "$DST"/*.zip

echo
echo "==> layout under $DST"
find "$DST" -maxdepth 2 -type d
for s in train val test; do
    echo "==> $s: $(ls "$DST/ReasonSeg/$s"/*.jpg 2>/dev/null | wc -l) images"
done
# expected: train 239, val 200, test 779
