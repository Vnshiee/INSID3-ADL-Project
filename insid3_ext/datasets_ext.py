"""Dataset construction: official builders + a COCO *tuning* split.

Test split: the official ``datasets.build_dataset`` is used unchanged.

Tune split (model selection): the paper selects hyper-parameters on the COCO-20i
*training* split (Supp. B). ``DatasetCOCOTune`` reuses the official ``DatasetCOCO``
episode sampler but reads ``splits/trn/fold*.pkl`` (train2014 images) and samples
from all 80 classes. Selecting our new hyper-parameters on these episodes keeps the
official test episodes untouched. ``image_overlap`` reports any image shared with
the test folds (expected: 0 — verify on the server, it is printed in the run log).
"""

from __future__ import annotations

import argparse
from os.path import join
from typing import Set

from datasets import build_dataset
from datasets.coco import DatasetCOCO


class DatasetCOCOTune(DatasetCOCO):
    """COCO train2014 episodes over all classes (mirrors DatasetCOCO.__init__)."""

    def __init__(self, datapath: str, shot: int, episodes: int = 1000):  # noqa: D107
        self.split = 'trn'
        self.fold = -1                       # official code path that merges the 4 fold files
        self.nfolds = 4
        self.nclass = 80
        self.test_episodes = episodes
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = 'train2014'
        self.base_path = join(datapath, 'COCO2014')
        self.class_ids = self.build_class_ids()                  # range(80)
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        # A class needs at least shot+1 images (target + distinct references).
        self.class_ids = [c for c in self.class_ids
                          if len(self.img_metadata_classwise.get(c, [])) > shot]
        if not self.class_ids:
            raise RuntimeError(f"No usable classes in {self.base_path}/splits/trn/*.pkl")
        self.img_metadata = self.build_img_metadata()


def make_args(cfg) -> argparse.Namespace:
    """Namespace with the attribute names the official dataset builders read."""
    return argparse.Namespace(data_root=cfg.data.data_root, fold=cfg.data.fold, shots=cfg.data.shots,
                              split=cfg.correspondence.split, dataset=cfg.data.dataset)


def build_eval_dataset(cfg):
    if cfg.data.split == "tune":
        return DatasetCOCOTune(cfg.data.data_root, cfg.data.shots, cfg.data.tune_episodes)
    return build_dataset(cfg.data.dataset, args=make_args(cfg))


def image_overlap(tune_ds: DatasetCOCOTune, data_root: str, shot: int) -> int:
    """Number of images shared between the tuning pool and all 4 COCO-20i test folds."""
    test_names: Set[str] = set()
    for f in range(4):
        test_names.update(DatasetCOCO(data_root, f, shot).img_metadata)
    return len(set(tune_ds.img_metadata) & test_names)
