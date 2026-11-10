#!/usr/bin/env bash
# Download / prepare evaluation datasets into $INSID3_DATA_ROOT, following the official
# docs/data.md commands verbatim. Nothing runs unless you name the datasets explicitly.
#
#   ./prepare_datasets.sh --dry-run coco spair          # print the commands only
#   ./prepare_datasets.sh coco pascal_part spair         # asks for confirmation per dataset
#   ./prepare_datasets.sh --yes all                      # everything (hundreds of GB, see below)
#
# Datasets: coco lvis paco_part pascal_part isic lung suim isaid permis spair
# Requirements: wget, curl, unzip, tar, git; gdown==5.2.0 (Google Drive); pandas (ISIC);
# Kaggle downloads (lung, suim) may require a Kaggle account / API token on some networks.
# Approximate download sizes are ESTIMATES (unverified) — check free disk space first.
set -euo pipefail
cd "$(dirname "$0")"
[[ -f env.sh ]] && source env.sh || source env.example.sh
DRY=0; YES=0; LIST=()
for a in "$@"; do case "$a" in --dry-run) DRY=1;; --yes) YES=1;; -h|--help) sed -n '2,13p' "$0"; exit 0;; *) LIST+=("$a");; esac; done
ALL=(coco lvis paco_part pascal_part isic lung suim isaid permis spair)
[[ ${#LIST[@]} -gt 0 ]] || { sed -n '2,13p' "$0"; exit 1; }
[[ "${LIST[0]}" == all ]] && LIST=("${ALL[@]}")

get_dataset_size() {
  case "$1" in
    coco) echo "~20 GB" ;;
    lvis) echo "~19 GB" ;;
    paco_part) echo "<1 GB (+LVIS images)" ;;
    pascal_part) echo "~1.5 GB" ;;
    isic) echo "~11 GB" ;;
    lung) echo "~4.5 GB" ;;
    suim) echo "<1 GB" ;;
    isaid) echo "unknown" ;;
    permis) echo "large (TAO test frames), unknown" ;;
    spair) echo "~0.3 GB" ;;
    *) echo "unknown" ;;
  esac
}
DATA="$INSID3_DATA_ROOT"; mkdir -p "$DATA"

run() { if [[ $DRY == 1 ]]; then echo "    $*"; else echo "+ $*"; eval "$@"; fi; }
need() { command -v "$1" >/dev/null || { echo "ERROR: '$1' not found ($2)"; exit 1; }; }

prep_coco() {
  run "mkdir -p '$DATA/COCO2014' && cd '$DATA/COCO2014'"
  run "wget -c http://images.cocodataset.org/zips/train2014.zip && wget -c http://images.cocodataset.org/zips/val2014.zip"
  run "unzip -q train2014.zip && unzip -q val2014.zip && rm -f train2014.zip val2014.zip"
  run "mkdir -p annotations && cd annotations && gdown --fuzzy https://drive.google.com/file/d/1cwup51kcr4m7v9jO14ArpxKMA4O3-Uge/view && gdown --fuzzy https://drive.google.com/file/d/1PNw4U3T2MhzAEBWGGgceXvYU3cZ7mJL1/view && unzip -q train2014.zip && unzip -q val2014.zip && rm -f *.zip && cd .."
  run "gdown --fuzzy https://drive.google.com/file/d/1K1UxhUzpjCe1jtgpsEUyrvU8gAlH1ZMy/view && unzip -q splits.zip && rm -f splits.zip"
}
prep_lvis() {
  run "mkdir -p '$DATA/LVIS/coco' && cd '$DATA/LVIS/coco'"
  run "wget -c http://images.cocodataset.org/zips/train2017.zip && wget -c http://images.cocodataset.org/zips/val2017.zip"
  run "unzip -q train2017.zip && unzip -q val2017.zip && rm -f *.zip && cd .."
  run "gdown --fuzzy https://drive.google.com/file/d/1itJC119ikrZyjHB9yienUPD0iqV12_9y/view && unzip -q lvis.zip && mv lvis/* . && rm -rf lvis lvis.zip"
}
prep_paco_part() {
  [[ -d "$DATA/LVIS/coco" || $DRY == 1 ]] || { echo "PACO-Part needs LVIS images first (prepare lvis)"; exit 1; }
  run "mkdir -p '$DATA/PACO-Part' && cd '$DATA/PACO-Part' && ln -sf ../LVIS/coco ."
  run "gdown --fuzzy https://drive.google.com/file/d/1VEXgHlYmPVMTVYd8RkT6-l8GGq0G9vHX/view && unzip -q paco.zip && rm -f paco.zip"
}
prep_pascal_part() {
  run "mkdir -p '$DATA/Pascal-Part' && cd '$DATA/Pascal-Part'"
  run "wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar && tar -xf VOCtrainval_03-May-2010.tar"
  run "gdown --fuzzy https://drive.google.com/file/d/1WaM0VM6I9b3u3v3w-QzFLJI8d3NRumTK/view && unzip -q pascal.zip && mv pascal/* VOCdevkit/VOC2010/ && rm -rf pascal.zip pascal VOCtrainval_03-May-2010.tar"
}
prep_isic() {
  run "mkdir -p '$DATA/ISIC' && cd '$DATA/ISIC'"
  run "wget -c https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip && wget -c https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip"
  run "unzip -q ISIC2018_Task1-2_Training_Input.zip && unzip -q ISIC2018_Task1_Training_GroundTruth.zip"
  run "wget -O ISIC_Split.py https://raw.githubusercontent.com/Matt-Su/DR-Adapter/main/data_util/ISIC_Split.py && mkdir -p isic && wget -O isic/class_id.csv https://raw.githubusercontent.com/Matt-Su/DR-Adapter/main/data_util/isic/class_id.csv"
  run "sed -i \"s|dir = '../data/ISIC/ISIC/ISIC2018_Task1-2_Training_Input/'|dir = 'ISIC2018_Task1-2_Training_Input/'|\" ISIC_Split.py"
  run "'$INSID3_PYTHON' ISIC_Split.py && rm -f *.zip ISIC_Split.py && rm -rf isic && rm -f ISIC2018_Task1-2_Training_Input/*.jpg"
}
prep_lung() {
  run "mkdir -p '$DATA/LungSegmentation' && cd '$DATA/LungSegmentation'"
  run "curl -L -o chest-xray-masks-and-labels.zip https://www.kaggle.com/api/v1/datasets/download/nikhilpandey360/chest-xray-masks-and-labels"
  run "unzip -q chest-xray-masks-and-labels.zip && mv 'Lung Segmentation'/* . && rm -rf 'Lung Segmentation' data *.zip"
}
prep_suim() {
  run "mkdir -p '$DATA/SUIM' && cd '$DATA/SUIM'"
  run "curl -L -o suim-merged.zip https://www.kaggle.com/api/v1/datasets/download/heyoujue/suim-merged"
  run "unzip -q suim-merged.zip && mv suim_merged/* . && rm -rf suim_merged suim-merged.zip"
}
prep_isaid() {
  run "mkdir -p '$DATA/iSAID' && cd '$DATA/iSAID'"
  run "gdown --fuzzy 'https://drive.google.com/file/d/17PQ1iKCbaj2OjwBdCn_VBh09ntI4lxgL/view?usp=sharing' -O iSAID_5i.zip && unzip -q iSAID_5i.zip && mv iSAID_patches/* ."
  run "gdown --fuzzy https://drive.google.com/file/d/1RBmVHXGhEXak2XB4OeCzQj2yswOmuMB2/view && unzip -q splits.zip && rm -rf *.zip iSAID_patches"
}
prep_permis() {
  run "mkdir -p '$DATA/PerMIRS' && cd '$DATA/PerMIRS'"
  run "wget -c https://motchallenge.net/data/3-TAO_TEST.zip && wget -c https://omnomnom.vision.rwth-aachen.de/data/BURST/annotations.zip"
  run "unzip -q 3-TAO_TEST.zip && unzip -q annotations.zip && git clone https://github.com/dvirsamuel/PDM.git"
  run "cd PDM/PerMIRS && sed -i 's|base_path = f\"/PerMIRS/{i}\"|base_path = f\"../../{i}\"|' permirs_gen_dataset.py"
  run "PYTHONPATH=.:.. '$INSID3_PYTHON' permirs_gen_dataset.py --images_base_dir ../../frames --annotations_file ../../test/all_classes.json && cd ../.."
  run "rm -rf PDM *.zip test train val frames info"
}
prep_spair() {
  run "mkdir -p '$DATA/SPair-71k' && cd '$DATA/SPair-71k'"
  run "wget -c https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz && tar -xzf SPair-71k.tar.gz --strip-components=1 && rm -f SPair-71k.tar.gz"
}

[[ $DRY == 1 ]] || { need wget "apt install wget"; need unzip "apt install unzip"; }
echo "Data root: $DATA  (free: $(df -h "$DATA" | awk 'NR==2{print $4}'))"
for ds in "${LIST[@]}"; do
  declare -F "prep_$ds" >/dev/null || { echo "unknown dataset: $ds (valid: ${ALL[*]})"; exit 1; }
  echo "== $ds  (download size $(get_dataset_size "$ds"))"
  if [[ $DRY == 0 && $YES == 0 ]]; then read -r -p "   prepare $ds now? [y/N] " ans; [[ "$ans" == y* ]] || continue; fi
  if [[ $DRY == 0 ]]; then case "$ds" in coco|lvis|paco_part|pascal_part|isaid) need gdown "pip install gdown==5.2.0";; esac; fi
  ( prep_"$ds" )      # subshell: each dataset starts from the repo directory
done
echo "Next: $INSID3_PYTHON scripts/check_env.py --datasets ${LIST[*]}"
