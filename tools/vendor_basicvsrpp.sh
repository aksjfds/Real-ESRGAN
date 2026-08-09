#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_REL="inference/weights/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth"
MODEL="$ROOT/$MODEL_REL"
URL="https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth"

if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
  echo "git-lfs is required to vendor the BasicVSR++ checkpoint." >&2
  exit 1
fi

mkdir -p "$(dirname "$MODEL")"
cd "$ROOT"
git lfs install --local
git lfs track "$MODEL_REL"

TMP="$MODEL.part"
rm -f "$TMP"
curl --fail --location --retry 3 --retry-delay 2 "$URL" --output "$TMP"
[ -s "$TMP" ] || { echo "Downloaded checkpoint is empty." >&2; exit 1; }
mv "$TMP" "$MODEL"

git add .gitattributes "$MODEL_REL"
echo "BasicVSR++ checkpoint staged through Git LFS: $MODEL_REL"
echo "Review with: git status && git lfs ls-files"
