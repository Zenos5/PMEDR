#!/usr/bin/env python
# coding: utf-8

import torch
from image2latex.model import Image2LatexModel
from data.dataset import LatexDataset, LatexPredictDataset, LatexSinglePredictDataset
from data.datamodule import DataModule
from image2latex.text import Text100k, Text170k, Text230k
import pytorch_lightning as pl
import argparse
import numpy as np
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training image2latex")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulate-batch", type=int, default=32)
    parser.add_argument("--data-path", type=str, help="data path")
    parser.add_argument("--img-path", type=str, help="image folder path")
    parser.add_argument(
        "--predict-img-path", type=str, help="image for predict path", default=None
    )
    parser.add_argument(
        "--dataset", type=str, help="choose dataset [100k, 170k]", default="100k"
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--log-text", action="store_true")
    parser.add_argument("--train-sample", type=int, default=5000)
    parser.add_argument("--val-sample", type=int, default=1000)
    parser.add_argument("--test-sample", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--log-step", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=12)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--enc-type", type=str, default="conv_row_encoder")
    # conv_row_encoder, conv_encoder, conv_bn_encoder, resnet_row_encoder
    parser.add_argument("--enc-dim", type=int, default=512)
    parser.add_argument("--emb-dim", type=int, default=80)
    parser.add_argument("--attn-dim", type=int, default=512)
    parser.add_argument("--dec-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--decode-type",
        type=str,
        default="greedy",
        help="Chose between [greedy, beamsearch]",
    )
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--model-name", type=str, default="conv_lstm")
    parser.add_argument("--grad-clip", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bayesian", type=bool, default=False)

    args = parser.parse_args()

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    torch.set_float32_matmul_precision('medium')

    text = None
    if args.dataset == "100k":
        text = Text100k()
    elif args.dataset == "170k":
        text = Text170k()
    elif args.dataset == "230k":
        text = Text230k()

    train_set = LatexDataset(
        data_path=args.data_path,
        img_path=args.img_path,
        data_type="train",
        n_sample=args.train_sample,
        dataset=args.dataset,
    )
    print(train_set, len(train_set))
    val_set = LatexDataset(
        data_path=args.data_path,
        img_path=args.img_path,
        data_type="validate",
        n_sample=args.val_sample,
        dataset=args.dataset,
    )
    test_set = LatexDataset(
        data_path=args.data_path,
        img_path=args.img_path,
        data_type="test",
        n_sample=args.test_sample,
        dataset=args.dataset,
    )
    # predict_set = LatexPredictDataset(data_path=args.data_path,
    #     img_path=args.img_path,
    #     n_sample=args.test_sample,
    #     dataset=args.dataset)
    # print(predict_set, len(predict_set))
    imfile = "../data/MSE/x0_103_x1_139_y0_263_y1_280_sharpened.jpg"
    # imfile = "../data/MSE/x0_200_x1_554_y0_593_y1_653_sharpened.jpg"
    print(imfile)
    # /home/aw742/PMEDR/data/MSE/x0_247_x1_475_y0_292_y1_337_sharpened.jpg

    predict_set = LatexSinglePredictDataset(imfile)
    print(predict_set, len(predict_set))
    

    steps_per_epoch = round(len(train_set) / args.batch_size)
    total_steps = steps_per_epoch * args.max_epochs
    dm = DataModule(
        train_set,
        val_set,
        test_set,
        predict_set,
        args.num_workers,
        args.batch_size,
        text,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Image2LatexModel(
        lr=args.lr,
        total_steps=total_steps,
        n_class=text.n_class,
        enc_dim=args.enc_dim,
        enc_type=args.enc_type,
        emb_dim=args.emb_dim,
        dec_dim=args.dec_dim,
        attn_dim=args.attn_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sos_id=text.sos_id,
        eos_id=text.eos_id,
        decode_type=args.decode_type,
        text=text,
        beam_width=args.beam_width,
        log_step=args.log_step,
        log_text=args.log_text,
        bayesian=args.bayesian,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    lr_stop = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min")

    accumulate_grad_batches = args.accumulate_batch // args.batch_size
    trainer = pl.Trainer(
        logger=None, #wandb_logger,
        callbacks=[lr_monitor, lr_stop],
        max_epochs=args.max_epochs,
        accelerator="gpu",
        strategy="ddp",
        log_every_n_steps=1,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=accumulate_grad_batches,
        devices=1,
    )

    ckpt_path = args.ckpt_path
    if ckpt_path:
        model = model.load_from_checkpoint(ckpt_path)

    if args.train:
        print("=" * 10 + "[Train]" + "=" * 10)
        trainer.fit(datamodule=dm, model=model, ckpt_path=ckpt_path)

    if args.val:
        print("=" * 10 + "[Validate]" + "=" * 10)
        trainer.validate(datamodule=dm, model=model, ckpt_path=ckpt_path)

    start_t = time.time()
    if args.test:
        print("=" * 10 + "[Test]" + "=" * 10)
        trainer.test(datamodule=dm, model=model, ckpt_path=ckpt_path)
    total_t = time.time() - start_t
    test_fps = (len(train_set)) / total_t
    print("Test FPS:", test_fps)

    if args.predict:
        print("=" * 10 + "[Predict]" + "=" * 10)
        latex = trainer.predict(datamodule=dm, model=model, ckpt_path=ckpt_path)
        print(latex)
        
