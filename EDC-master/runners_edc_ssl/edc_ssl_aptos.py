# runners_edc_ssl/edc_ssl_aptos.py
"""
SSL-EDC runner for APTOS dataset.
Identical to runners_edc/edc_aptos.py EXCEPT:
  - imports MoCo_R50_R50 instead of R50_R50
  - adds --moco_weights_path argument
  - adds --freeze_encoder argument
  - save_name defaults to 'edc_ssl_aptos'
"""

import os
import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils import get_logger, count_parameters, over_write_args_from_file
from train_utils import TBLog, get_optimizer_v2, get_multistep_schedule_with_warmup
from methods.edc1 import EDC
import shutil
from sklearn.metrics import confusion_matrix, classification_report

from datasets.dataset import AD_Dataset
from datasets.data_utils import get_data_loader
from models.edc_ssl import MoCo_R50_R50          # ← SSL model
import warnings
from configs.config_aptos import DATASET_DIR
from collections import Counter

warnings.filterwarnings("ignore")


def get_label(sample):
    if isinstance(sample, tuple) and len(sample) >= 2:
        label = sample[1]
    else:
        label = sample[-1]
    if isinstance(label, (np.ndarray, list)) and len(label) > 1:
        label = np.argmax(label)
    return int(label)


def main_worker(gpu, args):
    args.gpu = gpu
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    save_path = os.path.join(args.save_dir, args.save_name)
    logger = get_logger(args.save_name, save_path, "INFO")
    logger.warning(f"USE GPU: {args.gpu} for training")

    # ── Dataset & loaders ────────────────────────────────────────────────────
    train_dset = AD_Dataset(name=args.dataset, train=True,  data_dir=args.data_dir).get_dset()
    eval_dset  = AD_Dataset(name=args.dataset, train=False, data_dir=args.data_dir).get_dset()

    train_labels = np.array(train_dset.targets)
    eval_labels  = np.array(eval_dset.targets)
    train_counts = Counter(train_labels)
    eval_counts  = Counter(eval_labels)

    print(f"TrainSet: {len(train_dset)}  |  Normal: {train_counts[0]}  Abnormal: {train_counts.get(1,0)}")
    print(f"EvalSet : {len(eval_dset)}   |  Normal: {eval_counts[0]}   Abnormal: {eval_counts.get(1,0)}")

    generator_lb = torch.Generator()
    generator_lb.manual_seed(args.seed)

    loader_dict = {}
    loader_dict["train"] = get_data_loader(
        train_dset, args.batch_size,
        data_sampler=args.train_sampler,
        num_iters=args.num_train_iter,
        num_workers=args.num_workers,
        distributed=False,
        generator=generator_lb,
    )
    loader_dict["eval"] = get_data_loader(
        eval_dset, args.eval_batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # ── Model (SSL encoder) ───────────────────────────────────────────────────
    model = MoCo_R50_R50(
        moco_weights_path=args.moco_weights_path,
        img_size=args.img_size,
        train_encoder=True,
        stop_grad=True,
        reshape=True,
        bn_pretrain=False,
        freeze_encoder=args.freeze_encoder,
    )

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01

    runner = EDC(model=model, num_eval_iter=args.num_eval_iter, logger=logger)
    logger.info(f"Trainable params: {count_parameters(runner.model):,}")

    # ── Optimiser  (encoder gets lower lr) ───────────────────────────────────
    optimizer = get_optimizer_v2(
        runner.model, args.optim, args.lr, args.momentum,
        lr_encoder=args.lr_encoder, weight_decay=args.weight_decay,
    )
    scheduler = get_multistep_schedule_with_warmup(
        optimizer, milestones=[1e10], gamma=0.2, num_warmup_steps=0
    )
    runner.set_optimizer(optimizer, scheduler)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    runner.model = runner.model.to(device)
    args.device  = device

    logger.info(f"Arguments: {args}")
    runner.set_data_loader(loader_dict)

    if args.resume:
        runner.load_model(args.load_path)

    # ── Train ─────────────────────────────────────────────────────────────────
    runner.tb_log = TBLog(save_path, "tb", use_tensorboard=args.use_tensorboard)
    eval_dict = runner.train(args, device=device, logger=logger)
    logging.warning("Training and Evaluation COMPLETED!")

    # ── Final metrics summary ─────────────────────────────────────────────────
    metrics_table = pd.DataFrame({
        "Metric": ["AUC", "F1-score", "Accuracy", "Recall (Sensitivity)", "Specificity"],
        "Value":  [eval_dict["eval/AUC"], eval_dict["eval/f1"], eval_dict["eval/acc"],
                   eval_dict["eval/recall"], eval_dict["eval/specificity"]],
    })
    print("\n===== FINAL EVALUATION METRICS – SSL-EDC APTOS =====")
    print(metrics_table.to_string(index=False, float_format="%.4f"))

    y_true  = np.array(eval_dict["eval/y_true"])
    y_score = np.array(eval_dict["eval/y_score"])
    thr     = eval_dict["eval/best_thr"]
    y_pred  = (y_score >= thr).astype(int)

    cm     = confusion_matrix(y_true, y_pred)
    cm_df  = pd.DataFrame(cm, index=["Actual NORMAL","Actual ABNORMAL"],
                          columns=["Predicted NORMAL","Predicted ABNORMAL"])
    print("\n===== CONFUSION MATRIX =====")
    print(cm_df)
    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_true, y_pred,
                                target_names=["NORMAL","ABNORMAL"], digits=4))
    print(f"\nBest Threshold (F1-optimized): {thr:.4f}")


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes','true','t','y','1'): return True
    if v.lower() in ('no','false','f','n','0'): return False
    raise ValueError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SSL-EDC APTOS')

    # Saving
    parser.add_argument('--save_dir',  type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='edc_ssl_aptos')
    parser.add_argument('--resume',    action='store_true', default=False)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o','--overwrite', action='store_true', default=True)
    parser.add_argument('--use_tensorboard', action='store_true', default=True)

    # Training
    parser.add_argument('--epoch',           type=int, default=1)
    parser.add_argument('--num_train_iter',  type=int, default=1000)
    parser.add_argument('--num_eval_iter',   type=int, default=250)
    parser.add_argument('--batch_size',      type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)

    # Optimiser
    parser.add_argument('--optim',        type=str,   default='AdamW')
    parser.add_argument('--lr',           type=float, default=5e-4)
    parser.add_argument('--lr_encoder',   type=float, default=1e-5,
                        help='Lower lr for the SSL-pretrained encoder (fine-tune)')
    parser.add_argument('--momentum',     type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--amp',  type=str2bool, default=False)
    parser.add_argument('--clip', type=float,    default=1)

    # Data
    parser.add_argument('--data_dir',      type=str, default=DATASET_DIR)
    parser.add_argument('--dataset',       type=str, default='aptos')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--img_size',      type=int, default=256)
    parser.add_argument('--num_workers',   type=int, default=4)

    # SSL-specific
    parser.add_argument('--moco_weights_path', type=str,
                        required=True,
                        help='Path to moco_all5datasets_allN_200ep.pth')
    parser.add_argument('--freeze_encoder', type=str2bool, default=False,
                        help='If True, encoder weights are frozen (linear-probe mode)')

    # Misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu',  default='0', type=str)
    parser.add_argument('--c',    type=str, default='')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and not args.resume:
        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception(f'Model already exists: {save_path}')
    if args.resume and args.load_path is None:
        raise Exception('--load_path required when --resume is set')

    main_worker(int(args.gpu), args)
