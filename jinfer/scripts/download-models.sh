#!/usr/bin/env bash
# Downloads the jinfer test models (scripts/models.txt) into the models root, laid out as
# {source}/{user}/{repo}/{file}. Separate from the build on purpose - this is slow and large.
#
#   HF_TOKEN=hf_xxx scripts/download-models.sh [--only <substring>] [--list] [--root <dir>]
#
# Root: --root > $JINFER_MODELS > ../models next to the git checkout. Present files are
# skipped; hf.co files already in the HuggingFace hub cache ($HF_HOME) are symlinked instead
# of re-downloaded; interrupted downloads resume (curl -C -).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
manifest="$here/models.txt"
only=""
list=0
root="${JINFER_MODELS:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --only) only="$2"; shift 2 ;;
    --list) list=1; shift ;;
    --root) root="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$root" ]; then
  dir="$here"
  while [ "$dir" != "/" ] && [ ! -d "$dir/.git" ]; do dir="$(dirname "$dir")"; done
  [ -d "$dir/.git" ] || { echo "no git root found; pass --root or set JINFER_MODELS" >&2; exit 2; }
  root="$(dirname "$dir")/models"
fi

auth=()
[ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")

url_for() { # source user repo file
  case "$1" in
    hf.co)          echo "https://huggingface.co/$2/$3/resolve/main/$4" ;;
    modelscope.cn)  echo "https://modelscope.cn/models/$2/$3/resolve/master/$4" ;;
    *) echo "unknown source: $1" >&2; return 1 ;;
  esac
}

# A file already in the HuggingFace hub cache (hf download, jinfer pull, from_pretrained - all
# write it) is reused, not re-downloaded: ModelStore resolves through both layouts, the link
# just makes it visible in this tree too.
hub_cache="${HF_HOME:-$HOME/.cache/huggingface}/hub"
hub_cached() { # user repo file -> cached path on stdout, or status 1
  local repo="$hub_cache/models--$1--$2" revision candidate
  [ -f "$repo/refs/main" ] || return 1
  read -r revision < "$repo/refs/main"
  candidate="$repo/snapshots/$revision/$3"
  [ -e "$candidate" ] || return 1
  echo "$candidate"
}

grep -v -e '^#' -e '^[[:space:]]*$' "$manifest" | while read -r source user repo file; do
  case "$user/$repo/$file" in *"$only"*) ;; *) continue ;; esac
  dest="$root/$source/$user/$repo/$file"
  if [ "$list" = 1 ]; then
    printf '%-14s %s\n' "$([ -f "$dest" ] && echo present || echo missing)" "$dest"
    continue
  fi
  if [ -f "$dest" ]; then
    echo "present: $dest"
    continue
  fi
  if [ "$source" = hf.co ] && cached="$(hub_cached "$user" "$repo" "$file")"; then
    echo "link:    $dest -> $cached"
    mkdir -p "$(dirname "$dest")"
    ln -sf "$cached" "$dest"
    continue
  fi
  url="$(url_for "$source" "$user" "$repo" "$file")"
  echo "fetch:   $url"
  mkdir -p "$(dirname "$dest")"
  curl -fL --retry 3 -C - "${auth[@]}" -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
done
