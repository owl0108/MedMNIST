from argparse import ArgumentParser
from time import strftime

import lightning as pl
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from model import GeneralistModel
from dataset import MedMNISTDataModule

def main(hparams):
    pl.seed_everything(1)

    if hparams.tasks == ['all']:
        hparams.tasks = ['pathmnist', 'octmnist', 'pneumoniamnist', 'chestmnist', 'dermamnist', 'retinamnist',
                      'breastmnist', 'bloodmnist', 'organsmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist']
        
    print(f"Datasets to be used: {hparams.tasks}")

    model = GeneralistModel(hparams.tasks)
    data_module = MedMNISTDataModule(tasks=hparams.tasks,
                                     iter_mode=hparams.iter_mode,
                                     batch_size=hparams.batch_size,
                                     data_dir=hparams.data_dir,
                                     resize=True
                                     )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_ACC",
        dirpath="checkpoints",
        filename="best-{epoch:02d}-{val_ACC:.2f}",
        save_top_k=1,
        mode="max"
    )
    wandb_logger = WandbLogger(log_model='all', project='MedMNIST_lightning', name=hparams.run_name, config=hparams.__dict__)

    trainer = pl.Trainer(logger=wandb_logger,
                         fast_dev_run=hparams.fast_dev_runs,
                         callbacks=[checkpoint_callback],
                         max_epochs=hparams.max_epochs,
                         devices=hparams.devices,
                         accelerator=hparams.accelerator,
                         precision=hparams.precision,
                         num_sanity_val_steps=0,
                         log_every_n_steps=hparams.log_every_n_steps,
                         limit_train_batches=hparams.limit_train_batches,
                         limit_val_batches=hparams.limit_val_batches
                         )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iter_mode", default="max_size")
    parser.add_argument("--fast_dev_runs", type=int, default=False)
    parser.add_argument("--tasks", nargs='+', default=["all"])
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default=1)
    parser.add_argument("--max_epochs", default=None, type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--precision", default=32)
    parser.add_argument("--data_dir", default="/scratch/izar/ishii")
    parser.add_argument("--log_every_n_steps", default=50)
    parser.add_argument('--run_name', default=strftime("%m-%d-%Y_%H:%M:%S"), type=str, help='run_name for wandb')
    parser.add_argument('--limit_train_batches', default=None, type=int)
    parser.add_argument('--limit_val_batches', default=None, type=int)
    args = parser.parse_args()

    main(args)