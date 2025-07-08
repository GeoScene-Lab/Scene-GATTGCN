import argparse
import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import utils.callbacks
import utils.data
import utils.logging

DATA_PATHS = {
    "xinan": {
        "feat": "data/target.csv",
        "static_feat": "data/静态特征.csv",
        "adj": "data/adjacency_matrix_with_weights.csv",
        "target": "data/target.csv",
    },
}


def get_model(args, dm):
    model = None
    if args.model_name == "GATTGCN":
        model = models.GATTGCN(adj=dm.adj, input_dim=dm._feat.shape[2], hidden_dim=args.hidden_dim,num_heads=args.num_heads)
    return model


def get_task(args, model, dm):

    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, feat_max_val=dm.feat_max_val, **vars(args)
    )
    return task


def get_callbacks(args):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss")
    plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor="train_loss")
    callbacks = [
        checkpoint_callback,
        plot_validation_predictions_callback,
    ]
    return callbacks


def main_supervised(args):
    dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS[args.data]["feat"],
        static_feat_path=DATA_PATHS[args.data]["static_feat"],
        adj_path=DATA_PATHS[args.data]["adj"],
        target_path=DATA_PATHS[args.data]["target"],
        num_nodes=74,  # 站点个数
        num_dynamic_features=1,  # 每个站点特征个数
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        split_ratio=args.split_ratio,
        normalize=args.normalize,
    )
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    callbacks = get_callbacks(args)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, gpus=1, max_epochs=200)
    trainer.fit(task, dm)
    results = trainer.validate(datamodule=dm)
    return results


def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--data", type=str, help="The name of the dataset", choices=("xinan"), default="xinan"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GATTGCN"),
        default="GATTGCN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",),
        default="supervised",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")

    temp_args, _ = parser.parse_known_args()

    parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)

    args = parser.parse_args()
    utils.logging.format_logger(pl._logger)
    if args.log_path is not None:
        utils.logging.output_logger_to_file(pl._logger, args.log_path)

    try:
        results = main(args)
    except:  # noqa: E722
        traceback.print_exc()

