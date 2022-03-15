from typing import Optional

import pytorch_lightning as pl
import torch
from transformers import get_linear_schedule_with_warmup
from collections import Sequence

from src.wsd.sense_extractors import SenseExtractor

from src.utils.optimizers import RAdam
from src.utils.sift import AdversarialLearner, hook_sift_layer

import conf
from hparams import Hparams


class ModelModule(pl.LightningModule):
    def __init__(self, hps: Hparams, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hps = hps
        # self.save_hyperparameters(hps)
        self.sense_extractor: SenseExtractor = conf.SenseExtractorInst.instantiate()

        new_embedding_size = self.sense_extractor.model.config.vocab_size + 203
        self.sense_extractor.resize_token_embeddings(new_embedding_size)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_fn = self.get_adv_loss_fn() if self.hps.use_adversarial else self.get_default_loss_fn()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        relative_positions: Optional[torch.Tensor] = None,
        definitions_mask: Optional[torch.Tensor] = None,
        gold_markers: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> dict:

        sense_extractor_output = self.sense_extractor.extract(
            input_ids,
            attention_mask,
            padding_attention_mask,
            token_type_ids,
            relative_positions,
            definitions_mask,
            gold_markers,
        )

        output_dict = {
            "non_masked_logits": sense_extractor_output.non_masked_logits,
            "pred_logits": sense_extractor_output.prediction_logits,
            "pred_probs": sense_extractor_output.prediction_probs,
            "pred_markers": sense_extractor_output.prediction_markers,
            "gold_labels": sense_extractor_output.gold_labels,
            "attentions": sense_extractor_output.attentions,
        }

        return output_dict

    def get_adv_loss_fn(self):
        adv_modules = hook_sift_layer(
            self.sense_extractor,
            hidden_size=self.sense_extractor.model.config.hidden_size,
            learning_rate=self.hps.vat_learning_rate,
            init_perturbation=self.hps.vat_init_perturbation,
            target_module="model.embeddings.LayerNorm",  # hard code
        )
        adv = AdversarialLearner(self.sense_extractor, adv_modules)

        def adv_loss_fn(data, training=True):
            output = self.forward(**data)
            logits = output["pred_logits"]
            loss = self.criterion(logits, output["gold_labels"])
            if training:
                if isinstance(logits, Sequence):
                    logits = logits[-1]
                v_teacher = []

                t_logits = None
                if self.hps.vat_lambda > 0:

                    def pert_logits_fn(**data):
                        o = self.forward(**data)
                        logits = o["pred_logits"]
                        if isinstance(logits, Sequence):
                            logits = logits[-1]
                        return logits

                    loss += adv.loss(logits, pert_logits_fn, loss_fn=self.hps.vat_loss_fn, **data) * self.hps.vat_lambda

            return loss.mean()

        return adv_loss_fn

    def get_default_loss_fn(self):
        def default_loss_fn(data, *args, **kwargs):
            output = self.forward(**data)
            loss = self.criterion(output["pred_logits"], output["gold_labels"])
            return loss

        return default_loss_fn

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self.loss_fn(batch)
        self.log("loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss = self.loss_fn(batch, training=False)
        self.log(f"val_loss", loss, prog_bar=True)

    def get_optimizer_and_scheduler(self):

        no_decay = conf.no_decay_params

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": conf.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if conf.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                self.hps.lr,
            )
        elif conf.optimizer == "radam":
            optimizer = RAdam(
                optimizer_grouped_parameters,
                self.hps.lr,
            )
            return optimizer, None
        else:
            raise NotImplementedError

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=conf.num_warmup_steps,
            num_training_steps=conf.max_steps,
        )

        return optimizer, lr_scheduler

    def configure_optimizers(self):
        optimizer, lr_scheduler = self.get_optimizer_and_scheduler()
        if lr_scheduler is None:
            return optimizer
        return [optimizer], [{"interval": "step", "scheduler": lr_scheduler}]