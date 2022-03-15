import optuna
import argparse
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback

import conf
from hparams import Hparams
from debug import Debugger

from src.pl_data_modules import DataModule
from src.pl_model_modules import ModelModule

import src.scripts.raganato_evaluate as raganato_evaluate


def get_modules(args, hps: Hparams):
    datamodule = DataModule()
    model = ModelModule(hps)
    return dict(
        datamodule=datamodule,
        model=model,
    )


def get_trainer_config(hps: Hparams, trial: optuna.trial.Trial = None):
    trainer_config = conf.fixed_trainer_conf

    # callbacks
    trainer_config = trainer_config | dict(
        callbacks=[
            conf.RaganatoEvaluateCallBackInst.instantiate(),
            conf.EarlyStoppingCallBackInst.instantiate(),
            conf.CheckpointCallBackInst.instantiate(),
        ],
    )

    if trial is not None:
        trainer_config = trainer_config | dict(
            logger=False,
            deterministic=False,
            plugins=None,
        )
        trainer_config["callbacks"].append(PyTorchLightningPruningCallback(trial, monitor=conf.tune_metrics))
    return trainer_config


def tune(args, hps: Hparams):
    def _objective(trial: optuna.trial.Trial) -> float:
        hps.set_tune_config(trial)
        trainer = pl.Trainer(**get_trainer_config(hps, trial))
        trainer.fit(**get_modules(args, hps))
        return trainer.callback_metrics[conf.tune_metrics].item()

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction=conf.tune_direction, pruner=pruner)
    study.optimize(_objective, n_trials=conf.tune_n_trials)
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def main(args):

    # seed
    pl.seed_everything(conf.seed)

    # hparameters
    hps = Hparams()

    # debug
    if args.debugging:
        debugger = Debugger(hps)
        debugger.debug()
        exit()

    # tune
    if args.tuning:
        tune(args, hps)
        exit()

    # train, test
    trainer = pl.Trainer(**get_trainer_config(hps))
    if args.testing:
        # raganato evaluate
        raganato_evaluate.main()
        pass
    else:
        trainer.fit(**get_modules(args, hps))


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="96", type=int)
    parser.add_argument("-t", "--tuning", action="store_true")
    parser.add_argument("-p", "--pruning", action="store_true")
    parser.add_argument("-te", "--testing", action="store_true")
    parser.add_argument("-d", "--debugging", action="store_true")
    args = parser.parse_args()
    print(args)

    main(args)
