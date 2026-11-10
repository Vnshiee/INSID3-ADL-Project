#!/usr/bin/env bash
# Download DINOv3 weights. Meta distributes them behind a licence form: request access at
# https://github.com/facebookresearch/dinov3 and you receive per-user download URLs.
# Pass them via environment variables (never commit them):
#   DINOV3_URL_LARGE=... ./download_weights.sh large        (ViT-L/16, used for segmentation; ~1.2 GB)
#   DINOV3_URL_BASE=...  ./download_weights.sh base         (ViT-B/16, README correspondence setting; ~0.35 GB)
#   DINOV3_URL_SMALL=... ./download_weights.sh small
# Add --yes to skip the confirmation prompt. Files are saved with the official names in
# $INSID3_PRETRAIN_DIR and the sha256 prefix encoded in the name is verified.
set -euo pipefail
cd "$(dirname "$0")"
[[ -f env.sh ]] && source env.sh || source env.example.sh
YES=0; SIZES=()
for a in "$@"; do case "$a" in --yes) YES=1;; small|base|large) SIZES+=("$a");; *) echo "usage: $0 [small|base|large]... [--yes]"; exit 1;; esac; done
[[ ${#SIZES[@]} -gt 0 ]] || { sed -n '2,10p' "$0"; exit 1; }
get_weight_file() {
  case "$1" in
    small) echo "dinov3_vits16_pretrain_lvd1689m-08c60483.pth" ;;
    base)  echo "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" ;;
    large) echo "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" ;;
  esac
}
mkdir -p "$INSID3_PRETRAIN_DIR"
for s in "${SIZES[@]}"; do
  fname="$(get_weight_file "$s")"
  f="$INSID3_PRETRAIN_DIR/$fname"
  if [[ -f "$f" ]]; then echo "exists: $f"; else
    s_upper="$(echo "$s" | tr '[:lower:]' '[:upper:]')"
    var="DINOV3_URL_${s_upper}"; url="${!var:-}"
    [[ -n "$url" ]] || { echo "ERROR: set $var to the URL from Meta's licence e-mail"; exit 1; }
    if [[ $YES == 0 ]]; then read -r -p "Download $s weights to $f ? [y/N] " ans; [[ "$ans" == y* ]] || continue; fi
    curl -L --fail -o "$f.part" "$url" && mv "$f.part" "$f"
  fi
  want="$(basename "$f" .pth)"; want="${want##*-}"
  if command -v sha256sum >/dev/null; then
    got="$(sha256sum "$f" | cut -c1-${#want})"
  elif command -v shasum >/dev/null; then
    got="$(shasum -a 256 "$f" | cut -c1-${#want})"
  else
    got=""
  fi
  if [[ "$got" == "$want" ]]; then echo "ok: $f (sha256 prefix $got)"
  elif [[ -n "$got" ]]; then echo "WARNING: sha256 prefix of $f is $got, expected $want (corrupt or different release?)"; fi
done
