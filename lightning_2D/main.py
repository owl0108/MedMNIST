from time import strftime
import os


import torch
import clip
import lightning as pl
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from jsonargparse import ActionConfigFile, ArgumentParser

from generalist import GeneralistModel
from dataset import MedMNISTDataModule
from clip_text import MEDMNIST_DESCRIPTION

def main(hparams):
    #pl.seed_everything(1)
    if hparams.tasks == ['all']:
        hparams.tasks = ['pathmnist', 'octmnist', 'pneumoniamnist', 'chestmnist', 'dermamnist', 'retinamnist',
                      'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist']
        
    print(f"Datasets to be used: {hparams.tasks}")
    kwargs = hparams.__dict__

    if kwargs['get_clip_embedding']:
        if os.path.exists('./clip_encoding/clip_encoding_dict.pth'):
            clip_encoding = torch.load('./clip_encoding/clip_encoding_dict.pth')
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model, preprocess = clip.load("ViT-B/32", device=device)
            clip_encoding = {}
            for task in kwargs['tasks']:
                text = MEDMNIST_DESCRIPTION[task]
                tokens = clip.tokenize(text).to(device)
                with torch.no_grad():
                    text_features = clip_model.encode_text(tokens)
                    clip_encoding[task] =  text_features
            # create a new directory if it doensn`t exist
            os.makedirs('clip_encoding', exist_ok=True)
            torch.save(clip_encoding, './clip_encoding/clip_encoding_dict.pth')
        kwargs['clip_encoding'] = clip_encoding

    model = GeneralistModel(**kwargs)
    data_module = MedMNISTDataModule(**kwargs)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_ACC",
        dirpath="checkpoints",
        filename="best-{epoch:02d}-{val_ACC:.2f}",
        save_top_k=1,
        mode="max"
    )
    wandb_logger = WandbLogger(log_model=True, project='MedMNIST_lightning',
                               name=hparams.run_name+'_'+strftime("%m-%d-%Y_%H:%M:%S"),
                               config=hparams.__dict__)

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
                         limit_val_batches=hparams.limit_val_batches,
                         limit_test_batches=hparams.limit_test_batches,
                         detect_anomaly=hparams.detect_anomaly,
                         strategy=hparams.strategy,
                         num_nodes=hparams.num_nodes)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    parser = ArgumentParser()

    # add arguments
    parser = ArgumentParser()
    parser.add_argument("--iter_mode", default="max_size",
                        choices=["min_size", "max_size", "max_size_cycle", "sequential"])
    parser.add_argument("--fast_dev_runs", default=0, type=int)
    parser.add_argument("--tasks", nargs='+', default=["all"])
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default=1)
    parser.add_argument("--max_epochs", default=None, type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", default="/scratch/jed/ishii")
    parser.add_argument("--log_every_n_steps", default=10, type=int)
    parser.add_argument('--run_name', default='', type=str, help='run_name for wandb')
    parser.add_argument('--limit_train_batches', default=None, type=int)
    parser.add_argument('--limit_val_batches', default=None, type=int)
    parser.add_argument('--limit_test_batches', default=None, type=int)
    parser.add_argument('--precision', default='32', type=str)
    parser.add_argument('--head', default=None, type=str)
    parser.add_argument('--num_experts', default=20, type=int)
    parser.add_argument('--detect_anomaly', default=False, type=bool)
    parser.add_argument('--encoder_type', default='resnet18', type=str)
    parser.add_argument('--strategy', default='auto', type=str)
    parser.add_argument('--num_nodes', default=1, type=int)
    parser.add_argument('--pretrained', default=False, type=bool)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--num_head_layers', default=1, type=int)
    parser.add_argument('--combine_clip_embedding', default=False, type=bool)

    # Add argument to load config from a YAML file
    parser.add_argument('--config', action=ActionConfigFile)

    args = parser.parse_args()

    main(args)