import os
from logging import getLogger
from typing import Dict, Any

import pytorch_lightning as pl
import shutil
from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint

from src.scripts.raganato_evaluate import raganato_evaluate
from src.wsd.sense_inventories import SenseInventory, WordNetSenseInventory

from src.utils.conf_instance import Instance
from hparams import Hparams

logger = getLogger(__name__)


class ModelCheckpointWithBest(ModelCheckpoint):

    CHECKPOINT_NAME_BEST = "best.ckpt"

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if self.best_model_path == "":
            return
        orig_best = Path(self.best_model_path)
        shutil.copyfile(orig_best, orig_best.parent / self.CHECKPOINT_NAME_BEST)


class PredictorsRaganatoEvaluateCallback(pl.Callback):
    def __init__(
        self,
        raganato_path: str,
        test_raganato_path: str,
        wsd_framework_dir: str,
        samples_generator: Instance,
        test_samples_generator: Instance,
        predictors: Dict[str, Instance],
        wordnet_sense_inventory: SenseInventory,
        prediction_params: Dict[Any, Any],
    ):
        self.raganato_path = raganato_path
        self.test_raganato_path = test_raganato_path
        self.wsd_framework_dir = wsd_framework_dir
        self.samples_generator = samples_generator
        self.test_samples_generator = test_samples_generator
        self.predictors = {k: v.instantiate() for k, v in predictors.items()}
        self.wordnet_sense_inventory = wordnet_sense_inventory
        self.prediction_params = prediction_params

    def on_validation_epoch_start(self, trainer, pl_module):

        logger.info("PredictorsRaganatoEvaluateCallback started")

        pl_module.sense_extractor.evaluation_mode = True

        for predictor_name, predictor in self.predictors.items():

            logger.info(f"Doing {predictor_name}")

            # evaluate and log
            _, _, f1, _ = raganato_evaluate(
                raganato_path=self.raganato_path,
                wsd_framework_dir=self.wsd_framework_dir,
                module=pl_module,
                predictor=predictor,
                wordnet_sense_inventory=self.wordnet_sense_inventory,
                samples_generator=self.samples_generator,
                prediction_params=self.prediction_params,
            )

            # evaluate and log
            _, _, test_f1, _ = raganato_evaluate(
                raganato_path=self.test_raganato_path,
                wsd_framework_dir=self.wsd_framework_dir,
                module=pl_module,
                predictor=predictor,
                wordnet_sense_inventory=self.wordnet_sense_inventory,
                samples_generator=self.test_samples_generator,
                prediction_params=self.prediction_params,
            )

            pl_module.log(f"{predictor_name}_f1", f1, prog_bar=True, on_step=False, on_epoch=True)
            logger.info(f"{predictor_name}: {f1} f1")

            pl_module.log(
                f"{predictor_name}_test_f1",
                test_f1,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            logger.info(f"{predictor_name}: {test_f1} test_f1")

        pl_module.sense_extractor.evaluation_mode = False
