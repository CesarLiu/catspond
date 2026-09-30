"""Trains UniTraj's MTR in the WOMD setting of
responsibility/unitraj_configs/method/MTR_womd.yaml (the config the
responsibility adapter reads) -- UniTraj's train.py with the same trainer,
checkpointing and data pipeline, but importing MTR only: train.py imports
every UniTraj model and with them natten, torch_geometric and torch_cluster.
One difference: the training data are reshuffled every epoch (train.py
shuffles the sample list once).

Data are ScenarioNet directories (scenarionet.convert_waymo). UniTraj first
preprocesses every scenario of a directory into an h5 cache under
--cache-path/<parent name>/<directory name> (uncompressed, about 1.8 MB per
predicted agent with 768 map polylines), then trains from it; train and
validation directories therefore need different names. Re-running reuses the
cache.

Example (server, UniTraj environment, from the cat repository root):
    python -m scripts.responsibility.train_unitraj --exp-name mtr_womd \\
        --train-data /data/womd_sn/training --val-data /data/womd_sn/validation \\
        --cache-path /data/unitraj_cache --out-dir /data/unitraj_ckpt --devices 0
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from responsibility.unitraj import (  # noqa: E402
    DEFAULT_METHOD,
    import_real_scenarionet,
    import_unitraj,
    load_config,
    unitraj_root,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-data", nargs="+", required=True, help="ScenarioNet directories.")
    p.add_argument("--val-data", nargs="+", required=True)
    p.add_argument("--exp-name", default="mtr_womd")
    p.add_argument("--out-dir", default="unitraj_ckpt", help="Checkpoints go to OUT/<exp-name>/.")
    p.add_argument("--cache-path", default="unitraj_cache")
    p.add_argument("--devices", type=int, nargs="+", default=[0])
    p.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    p.add_argument("--max-data-num", type=int, default=None, help="Training samples used per directory.")
    p.add_argument("--batch-size", type=int, default=None, help="Total over devices (default: the config's).")
    p.add_argument("--epochs", type=int, default=None, help="Default: the config's max_epochs (40).")
    p.add_argument("--resume", default=None, help="A checkpoint to continue from.")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (default: CSV logs).")
    p.add_argument("--method", default=DEFAULT_METHOD)
    p.add_argument("--unitraj-root", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    import_real_scenarionet()  # before UniTraj: reading datasets needs the installed ones
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader

    torch.set_float32_matmul_precision("medium")
    abs_paths = lambda ps: [str(Path(x).resolve()) for x in ps]  # noqa: E731
    config = load_config(method=args.method, root=args.unitraj_root)
    config.update(
        exp_name=args.exp_name, debug=False, devices=list(args.devices), seed=args.seed,
        train_data_path=abs_paths(args.train_data), val_data_path=abs_paths(args.val_data),
        max_data_num=[args.max_data_num] * len(args.train_data), starting_frame=[0] * len(args.train_data),
        cache_path=str(Path(args.cache_path).resolve()), load_num_workers=args.workers,
    )
    if args.epochs is not None:
        config["max_epochs"] = config["method"]["max_epochs"] = args.epochs
    if args.batch_size is not None:
        config["train_batch_size"] = config["method"]["train_batch_size"] = args.batch_size
    out_dir = Path(args.out_dir).resolve() / args.exp_name
    resume = str(Path(args.resume).resolve()) if args.resume else None

    os.chdir(unitraj_root(args.unitraj_root) / "unitraj")  # MTR's intention points file is relative to it
    utils = import_unitraj("utils.utils", args.unitraj_root)
    mtr = import_unitraj("models.mtr.MTR", args.unitraj_root)
    datasets = import_unitraj("datasets.MTR_dataset", args.unitraj_root)
    utils.set_seed(args.seed)

    model = mtr.MotionTransformer(config)
    train_set = datasets.MTRDataset(config=config, is_validation=False)
    val_set = datasets.MTRDataset(config=config, is_validation=True)
    n_dev = len(args.devices)
    train_loader = DataLoader(train_set, batch_size=max(config["method"]["train_batch_size"] // n_dev, 1),
                              num_workers=args.workers, drop_last=False, shuffle=True,
                              collate_fn=train_set.collate_fn)
    val_loader = DataLoader(val_set, batch_size=max(config["method"]["eval_batch_size"] // n_dev, 1),
                            num_workers=args.workers, shuffle=False, drop_last=False, collate_fn=train_set.collate_fn)
    # flat names (UniTraj's "{epoch}-{val/brier_fde:.2f}" puts a directory into them): epoch12-brier_fde1.84.ckpt
    checkpoint = ModelCheckpoint(monitor="val/brier_fde", filename="epoch{epoch}-brier_fde{val/brier_fde:.2f}",
                                 auto_insert_metric_name=False, save_top_k=1, save_last=True, mode="min",
                                 dirpath=str(out_dir))
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project="unitraj", name=args.exp_name, id=args.exp_name)
    else:
        from pytorch_lightning.loggers import CSVLogger

        logger = CSVLogger(str(out_dir), name="logs")
    trainer = pl.Trainer(
        max_epochs=config["method"]["max_epochs"], logger=logger, devices=list(args.devices),
        gradient_clip_val=config["method"]["grad_clip_norm"], accelerator="gpu", profiler="simple",
        strategy="ddp" if n_dev > 1 else "auto", callbacks=[checkpoint],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume)
    print(f"best checkpoint: {checkpoint.best_model_path}\nlast checkpoint: {checkpoint.last_model_path}")


if __name__ == "__main__":
    main()
