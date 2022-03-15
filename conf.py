from typing import Union, List, Dict, Optional
from src.utils.conf_instance import Instance
from src.wsd.sense_inventories import SenseInventory, WordNetSenseInventory
from src.wsd.tokenizer import DeBERTaTokenizer
from src.wsd.dependency_finder import (
    DependencyFinder,
    PPMIPolysemyDependencyFinder,
)
from src.wsd.disambiguation_corpora import DisambiguationCorpus, WordNetCorpus
from src.wsd.dataset import (
    ConanDataset,
    build_samples_generator_from_disambiguation_corpus,
)
from src.wsd.sense_extractors import (
    SenseExtractor,
    DebertaPositionalExtractor,
)
from src.wsd.continuous_predict import (
    Predictor,
    TeacherForcedPredictor,
    GreedyDepPredictor,
    BeamDepPredictor,
)
from src.pl_callbacks import PredictorsRaganatoEvaluateCallback, ModelCheckpointWithBest
from pytorch_lightning.callbacks import EarlyStopping


# == training == ##########################
# reproducibility
seed: int = 96
deterministic: bool = False

# cpu, gpu, tpu
gpus: Union[int, None] = 1  # if gpus == None & tpu_cores == None, then use cpu
tpu_cores: Union[int, None] = None

# epoch or step
max_epochs: Union[int, None] = None
max_steps: Union[int, None] = 1_000_000 if max_epochs is None else None
val_check_interval: int = 2000 if max_epochs is None else None

# batch
tokens_per_batch: int = 1536
max_batch_size: int = 128
section_size: int = 10_000

# gradient
accumulate_grad_batches: int = 8
gradient_clip_val: float = 1.0
no_decay_params: List[str] = [
    "bias",
    "LayerNorm.weight",
]
weight_decay: float = 0.01
"""
adamw
radam
"""
optimizer: str = "radam"  # ["adamw", "radam"]
lr_schduler: str = ""  # ["", "linear"]
num_warmup_steps: int = 10000  # require edit

# computing precision
precision: int = 16
amp_level: str = "O1"

# pretrained model
pmodel: Dict = {
    "name": "microsoft/deberta-large",
    "class": None,
    "use_special_tokens": True,
    "max_len": 4096,
    "output_attentions": False,
}
if pmodel["output_attentions"]:
    print("###############################")
    print("  output_attentions is True")
    print("###############################")

# dataset (主に訓練時)
sentence_window: int = 2
randomize_sentence_window: bool = True
remove_multilabel_instances: bool = True
shuffle_definitions: bool = True
randomize_dependencies: bool = True
sense_frequencies_path: Optional[str] = None
num_workers: int = 0
text_encoding_strategy: str = "relative-positions"

# callbacks
callbacks_monitor: str = "greedy_ppmi_f1"
callbacks_mode: str = "max"

# == tuning == #############################
tune_metrics: str = "val_loss"
tune_direction: str = "minimize"
tune_n_trials: int = 20

# == Path == #############################
path_wsd_framework_dir = "corpus/WSD_Evaluation_Framework/"
path_sense_inventory = path_wsd_framework_dir + "Data_Validation/candidatesWN30.txt"
path_raganato_trains = [
    path_wsd_framework_dir + "Training_Corpora/SemCor/semcor",
    # path_wsd_framework_dir + "Training_Corpora/WNGC/wngc",
    # path_wsd_framework_dir + "Training_Corpora/WNEX/wnex",
]
path_raganato_validations = [
    path_wsd_framework_dir + "Evaluation_Datasets/semeval2007/semeval2007",
]
path_raganato_tests = [
    path_wsd_framework_dir + "Evaluation_Datasets/ALL/ALL",
]
path_single_counter = "corpus/pmi/lemma_counter.txt"
path_pair_counter = "corpus/pmi/pairs_counter.txt"

# checkpoint
path_checkpoint_4_test = "checkpoints/best.ckpt"

# == trainer == #############################
fixed_trainer_conf = dict(
    # max_epochs=max_epochs,
    max_steps=max_steps,
    val_check_interval=val_check_interval,
    gradient_clip_val=gradient_clip_val,
    gpus=gpus,
    tpu_cores=tpu_cores,
    deterministic=deterministic,
    accumulate_grad_batches=accumulate_grad_batches,
    precision=precision,
    amp_level=amp_level,
)

# == Instance == #############################
# inventory
class InventoryInst(Instance):
    def __init__(self):
        super().__init__(
            target=WordNetSenseInventory,
            wn_candidates_path=path_sense_inventory,
            corpora_names=path_raganato_trains,
        )


inventory = InventoryInst.instantiate()

# tokenizer
class TokenizerInst(Instance):
    def __init__(self):
        super().__init__(
            target=DeBERTaTokenizer,
            transformer_model=pmodel["name"],
            target_marker=["<d>", "</d>"],
            context_definitions_token="CONTEXT_DEFS",
            context_markers={
                "number": 1,
                "pattern": ["DEF_SEP", "DEF_END"],
            },
            add_prefix_space=True,
        )


# dependency finder
class DependencyFinderInst(Instance):
    def __init__(self):
        super().__init__(
            target=PPMIPolysemyDependencyFinder,
            sense_inventory=inventory,
            single_counter_path=path_single_counter,
            pair_counter_path=path_pair_counter,
            energy=0.7,
            max_dependencies=9,
            normalize_ppmi=True,
            minimum_ppmi=0.1,
            undirected=False,
            with_pos=True,
        )


dependency_finder = DependencyFinderInst.instantiate()

# corpus
class TrainCorpusInst(Instance):
    def __init__(self):
        super().__init__(
            target=[
                (
                    WordNetCorpus,
                    dict(
                        raganato_path=raganato_path,
                    ),
                )
                for raganato_path in path_raganato_trains
            ],
            materialize=True,
            cached=False,
            shuffle=False,
        )


class ValidationCorpusInst(Instance):
    def __init__(self):
        super().__init__(
            target=[
                (
                    WordNetCorpus,
                    dict(
                        raganato_path=raganato_path,
                    ),
                )
                for raganato_path in path_raganato_validations
            ],
            materialize=True,
            cached=False,
            shuffle=False,
        )


class TestCorpusInst(Instance):
    def __init__(self):
        super().__init__(
            target=[
                (
                    WordNetCorpus,
                    dict(
                        raganato_path=raganato_path,
                    ),
                )
                for raganato_path in path_raganato_tests
            ],
            materialize=True,
            cached=False,
            shuffle=False,
        )


# dataset
dataset_common_kwargs = dict(
    target=ConanDataset.from_disambiguation_corpus,
    sense_inventory=inventory,
    dependency_finder=dependency_finder,
    tokenizer=TokenizerInst,
    sentence_window=sentence_window,
    randomize_sentence_window=randomize_sentence_window,
    remove_multilabel_instances=remove_multilabel_instances,
    shuffle_definitions=shuffle_definitions,
    randomize_dependencies=randomize_dependencies,
    sense_frequencies_path=sense_frequencies_path,
    text_encoding_strategy=text_encoding_strategy,
    use_definition_start=True,
    # basedataset
    tokens_per_batch=tokens_per_batch,
    max_batch_size=max_batch_size,
    section_size=section_size,
    prebatch=True,
    max_length=pmodel["max_len"],
)


class TrainDatasetInst(Instance):
    def __init__(self):
        super().__init__(
            **(
                dataset_common_kwargs
                | dict(
                    disambiguation_corpus=TrainCorpusInst,
                    shuffle=True,
                )
            )
        )


class ValidationDatasetInst(Instance):
    def __init__(self):
        super().__init__(
            **(
                dataset_common_kwargs
                | dict(
                    disambiguation_corpus=ValidationCorpusInst,
                    shuffle=False,
                    randomize_sentence_window=False,
                    remove_multilabel_instances=False,
                    shuffle_definitions=False,
                    randomize_dependencies=False,
                )
            )
        )


# sense extractor
class SenseExtractorInst(Instance):
    def __init__(self):
        super().__init__(
            target=DebertaPositionalExtractor,
            transformer_model=pmodel["name"],
            dropout=0.0,
            use_definitions_mask=True,
            output_attentions=pmodel["output_attentions"],
        )


# predictor
class GreedyDepPredictorInst(Instance):
    def __init__(self):
        super().__init__(
            target=GreedyDepPredictor,
            dependency_finder=dependency_finder,
        )


# callbacks
class TrainSampleGeneratorInst(Instance):
    def __init__(self):
        super().__init__(
            target=build_samples_generator_from_disambiguation_corpus,
            disambiguation_corpus=TrainCorpusInst,
            sense_inventory=inventory,
            dependency_finder=dependency_finder,
            sentence_window=2,
            randomize_sentence_window=False,
            remove_multilabel_instances=False,
            shuffle_definitions=False,
            randomize_dependencies=False,
        )


class EvalSampleGeneratorInst4RaganatoEvaluateCallBackInst(Instance):
    def __init__(self):
        super().__init__(
            target=build_samples_generator_from_disambiguation_corpus,
            disambiguation_corpus=ValidationCorpusInst,
            sense_inventory=inventory,
            dependency_finder=dependency_finder,
            sentence_window=2,
            randomize_sentence_window=False,
            remove_multilabel_instances=False,
            shuffle_definitions=False,
            randomize_dependencies=False,
        )


class TestSampleGeneratorInst4RaganatoEvaluateCallBackInst(Instance):
    def __init__(self):
        super().__init__(
            target=build_samples_generator_from_disambiguation_corpus,
            disambiguation_corpus=TestCorpusInst,
            sense_inventory=inventory,
            dependency_finder=dependency_finder,
            sentence_window=sentence_window,
            randomize_sentence_window=False,
            remove_multilabel_instances=False,
            shuffle_definitions=False,
            randomize_dependencies=False,
        )


class RaganatoEvaluateCallBackInst(Instance):
    def __init__(self):
        super().__init__(
            target=PredictorsRaganatoEvaluateCallback,
            raganato_path=path_raganato_validations[0],
            test_raganato_path=path_raganato_tests[0],
            wsd_framework_dir=path_wsd_framework_dir,
            samples_generator=EvalSampleGeneratorInst4RaganatoEvaluateCallBackInst,
            test_samples_generator=TestSampleGeneratorInst4RaganatoEvaluateCallBackInst,
            predictors={
                "greedy_ppmi": GreedyDepPredictorInst,
            },
            wordnet_sense_inventory=inventory,
            prediction_params={
                "text_encoding_strategy": "relative-positions",
                "token_batch_size": 4096,
                "progress_bar": False,
            },
        )


class EarlyStoppingCallBackInst(Instance):
    def __init__(self):
        super().__init__(
            target=ModelCheckpointWithBest,
            monitor=callbacks_monitor,
            mode=callbacks_mode,
            verbose=True,
            save_top_k=3,
            dirpath="checkpoints",
        )


class CheckpointCallBackInst(Instance):
    def __init__(self):
        super().__init__(
            target=EarlyStopping,
            monitor=callbacks_monitor,
            mode=callbacks_mode,
            patience=10000,
        )


# == filter warnings ###########################
import warnings, re, logging

warnings.filterwarnings("ignore")


def _set_global_logging_level(level=logging.ERROR, prefices=[""]):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


_set_global_logging_level(
    logging.ERROR,
    ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"],
)
