import conf
from hparams import Hparams

from typing import Iterator, Tuple, List, Optional

import torch
import os
import shutil

# import torchvision
# from torch.utils.tensorboard import SummaryWriter
from transformers import DebertaTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from src.pl_model_modules import ModelModule
from src.wsd.tokenizer import DeBERTaTokenizer as ConanDeBERTaTokenizer
from src.wsd.dataset import ConanDataset, ConanSample
from src.wsd.sense_inventories import WordNetSenseInventory

from collections import defaultdict
from nltk.corpus import wordnet
import dill
import os
import random


class Debugger:
    def __init__(self, hps: Hparams):
        self.device = "cuda:0"
        conf.pmodel["output_attentions"] = True  # hard code
        self.module = ModelModule(hps=hps)
        self.module.to(torch.device(self.device))
        self.module.freeze()
        self.module.sense_extractor.evaluation_mode = True
        self._tokenizer: ConanDeBERTaTokenizer = conf.TokenizerInst.instantiate()
        self.tokenizer: DebertaTokenizer = self._tokenizer.tokenizer
        self.inventory: WordNetSenseInventory = conf.inventory
        self.samples = conf.EvalSampleGeneratorInst4RaganatoEvaluateCallBackInst.instantiate()
        self.tgt_sample_ids = None  # None is all sample
        self.d_token_ids = [50265, 50266]
        self.file_name = "cls_sep"

        x = self.inventory.get_definition()
        a = 0

    def mat2heatmap(
        self,
        mat,
        x_labels: List,
        y_labels: List,
        line_positions: List,
        title: str = "Title",
        xtick_step: int = 1,
        ytick_step: int = 1,
        output_dir="checkpoints",
        file_name="figure",
        cmap="plasma",
    ):

        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xticks(
            np.arange(0, len(x_labels), xtick_step),
            x_labels[::xtick_step],
            rotation=90,
            fontsize=7,
        )
        ax.set_yticks(
            np.arange(0, len(y_labels), ytick_step),
            y_labels[::ytick_step],
            fontsize=7,
        )
        ax.hlines(
            line_positions,
            0,
            len(y_labels) - 1,
            "white",
            linestyles="dotted",
            linewidth=1.0,
        )
        ax.vlines(
            line_positions,
            0,
            len(y_labels) - 1,
            "white",
            linestyles="dotted",
            linewidth=1.0,
        )
        # ax.imshow(mat, cmap="hot")
        aximg = ax.imshow(mat, cmap=cmap)
        fig.colorbar(aximg, ax=ax)
        plt.savefig(f"{output_dir}/{file_name}.png", format="png", dpi=220)
        plt.savefig(f"{output_dir}/x_{file_name}.eps", format="eps", dpi=220)
        print("(ENTER):")
        input_text = input()
        if input_text == "break":
            return True

    def create_examples(self):

        self.train_samples = conf.TrainSampleGeneratorInst.instantiate()

        completed_synid2example = defaultdict(list)

        synid2example = defaultdict(list)
        for tr_sample in tqdm(self.train_samples(), desc="train"):
            tr_sample: ConanSample = tr_sample
            text = " ".join([di.text for di in tr_sample.kwargs["original_disambiguation_context"]])
            labels = tr_sample.disambiguation_instance.labels
            instance_id = tr_sample.disambiguation_instance.instance_id
            synsets = [wordnet.lemma_from_key(sense).synset() for sense in labels]
            synids = [self.inventory.get_synid(synset) for synset in synsets]
            for synid in synids:
                synid2example[synid].append((instance_id, text))

        for synset in tqdm(wordnet.all_synsets(), desc="wordnet"):
            synid = self.inventory.get_synid(synset)
            examples = synset.examples()
            for example in examples:
                synid2example[synid].append((None, example))

        for synid, examples in tqdm(synid2example.items(), desc="all examples"):
            if synid not in completed_synid2example:
                examples = random.sample(examples, len(examples))
                examples_texts = [item[1] for item in examples if item is not None]
                examples_texts_instance_ids = [item[0] for item in examples if item is not None]
                completed_synid2example[synid].extend(list(zip(examples_texts, examples_texts_instance_ids)))

        synid2example_path = "examples/b_synid2example.pkl"
        with open(synid2example_path, "wb") as f:
            dill.dump(completed_synid2example, f)
        print("dumped examples/b_synid2example.pkl")
        exit()

    def debug(self):

        # self.create_examples()

        dataset = ConanDataset.from_samples(
            list(self.samples()),
            tokenizer=self._tokenizer,
            use_definition_start=True,
            text_encoding_strategy=conf.text_encoding_strategy,
            tokens_per_batch=1536,
            max_batch_size=128,
            section_size=2_000,
            prebatch=True,
            shuffle=False,
            max_length=self._tokenizer.model_max_length,
        )
        dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

        # predict
        iterator = dataloader

        for batch_idx, batch in enumerate(iterator):

            batch_samples: List[ConanSample] = batch["original_sample"]
            batch_definitions_positions = batch["definitions_positions"]

            sample_ids_in_batch = [sample.sample_id for sample in batch_samples]
            for sample_id in sample_ids_in_batch:
                if self.tgt_sample_ids is None or sample_id in self.tgt_sample_ids:
                    break
            else:
                continue

            with autocast(enabled=True):
                with torch.no_grad():
                    batch_out = self.module(**{k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()})
                    input_ids = batch["input_ids"]
                    trancated_input_ids = []
                    instance_possible_definitions = batch["instance_possible_definitions"]
                    decoded_texts = []
                    definition_positions = batch["definitions_positions"]
                    gold_definitions = batch["gold_definitions"]
                    for ids in input_ids:
                        ids = ids[ids != self.tokenizer.pad_token_id]
                        ids = ids.cpu().tolist()
                        decoded_texts.append([self.tokenizer.decode([encoded_id]) for encoded_id in ids])
                        trancated_input_ids.append(ids)
                    attentions = batch_out["attentions"]
                    attentions = torch.mean(attentions, dim=2)  # mean heads
                    attentions = torch.clamp(attentions, min=0.0, max=0.25)
                    attentions = attentions * 4
                    for data_idx, (
                        attn_p_batch,
                        decoded_text,
                        b_sample,
                        mask_position,
                        possible_definitions,
                        trancated_ids,
                        gold_definition,
                    ) in enumerate(
                        zip(
                            attentions,
                            decoded_texts,
                            batch_samples,
                            definition_positions,
                            instance_possible_definitions,
                            trancated_input_ids,
                            gold_definitions,
                        )
                    ):
                        if (self.tgt_sample_ids is None or b_sample.sample_id in self.tgt_sample_ids) and len(b_sample.candidate_definitions) > 1:
                            output_dir = f"checkpoints/{batch_idx}_{data_idx}"
                            if os.path.exists(output_dir):
                                shutil.rmtree(output_dir)
                            os.makedirs(output_dir)
                            with open(f"{output_dir}/original_sentense.txt", "w") as f:
                                print(" ".join(decoded_text).replace("  ", " "), file=f)
                                print(gold_definition, file=f)
                            tgt_word_position = [idx for idx, _id in enumerate(trancated_ids) if _id in self.d_token_ids]
                            for layer_idx, attn_p_layer in enumerate(attn_p_batch):
                                attn_mat = attn_p_layer.detach().cpu().numpy()
                                attn_mat = attn_mat[: len(decoded_text), : len(decoded_text)]
                                title = f"{batch_idx}_{layer_idx}"
                                input_text = self.mat2heatmap(
                                    attn_mat,
                                    title="",
                                    # line_positions=[0] + mask_position,
                                    line_positions=mask_position,
                                    x_labels=decoded_text,
                                    y_labels=decoded_text,
                                    xtick_step=3,
                                    ytick_step=3,
                                    output_dir=output_dir,
                                    file_name=f"{layer_idx}",
                                )

                                if input_text:
                                    break
