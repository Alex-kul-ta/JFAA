# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import logging
import pprint
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    ShardingStrategy,
)
from torch.distributed.fsdp import MixedPrecision

from evals.action_anticipation_frozen.dataloader import filter_annotations, init_data
from evals.action_anticipation_frozen.losses import sigmoid_focal_loss
from evals.action_anticipation_frozen.metrics import ClassMeanRecall
from evals.action_anticipation_frozen.models import init_classifier, init_module
from evals.action_anticipation_frozen.utils import init_opt
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.cuda.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def _metric_to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _summarize_probe_head_metrics(metrics):
    accuracies = [_metric_to_float(m["accuracy"]) for m in metrics]
    recalls = [_metric_to_float(m["recall"]) for m in metrics]
    best_accuracy_head = max(range(len(accuracies)), key=lambda i: accuracies[i])
    best_recall_head = max(range(len(recalls)), key=lambda i: recalls[i])
    return dict(
        accuracy=accuracies[best_accuracy_head],
        recall=recalls[best_recall_head],
        best_head_by_accuracy=int(best_accuracy_head),
        best_head_by_recall=int(best_recall_head),
        per_head=[
            dict(head=int(i), accuracy=float(accuracy), recall=float(recall))
            for i, (accuracy, recall) in enumerate(zip(accuracies, recalls))
        ],
    )


def main(args_eval, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    val_only = args_eval.get("val_only", False)
    if val_only:
        logger.info("VAL ONLY")

    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    val_only = args_eval.get("val_only", False)
    eval_tag = args_eval.get("tag", None)

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")

    # -- CLASSIFIER
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads")

    # -- DATA
    args_data = args_exp.get("data")
    dataset = args_data.get("dataset")
    base_path = args_data.get("base_path")
    file_format = args_data.get("file_format", 1)  # used only for video mode
    input_format = args_data.get("input_format", "video")
    video_info_path = args_data.get("video_info_path", None)
    frame_template = args_data.get("frame_template", "frame_{:010d}.jpg")
    frame_index_offset = args_data.get("frame_index_offset", 1)

    num_workers = args_data.get("num_workers", 12)
    pin_mem = args_data.get("pin_memory", True)

    frames_per_clip = args_data.get("frames_per_clip")
    frames_per_second = args_data.get("frames_per_second")
    resolution = args_data.get("resolution", 224)

    train_anticipation_time_sec = args_data.get("train_anticipation_time_sec")
    train_anticipation_point = args_data.get("train_anticipation_point")
    val_anticipation_point = args_data.get("val_anticipation_point", [1.0, 1.0])
    val_anticipation_time_sec = args_data.get("anticipation_time_sec")

    auto_augment = args_data.get("auto_augment")
    motion_shift = args_data.get("motion_shift")
    reprob = args_data.get("reprob")
    random_resize_scale = args_data.get("random_resize_scale")

    train_annotations_path = args_data.get("dataset_train")
    val_annotations_path = args_data.get("dataset_val")
    train_data_path = base_path
    val_data_path = base_path

    if input_format.lower() == "frames" and not video_info_path:
        raise ValueError(
            "When `input_format: frames`, you must provide `video_info_path` "
            "(e.g. EPIC_100_video_info.csv)."
        )

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    use_focal_loss = args_opt.get("use_focal_loss", False)
    criterion = sigmoid_focal_loss if use_focal_loss else torch.nn.CrossEntropyLoss()
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]
    use_fsdp_classifier = bool(args_opt.get("use_fsdp_classifier", True))

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    logger.info(f"use_fsdp_classifier={use_fsdp_classifier}")

    folder = os.path.join(pretrain_folder, "action_anticipation_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    log_txt = os.path.join(folder, "log.txt")
    latest_path = os.path.join(folder, "latest.pt")
    best_path = os.path.join(folder, "best.pt")
    best_metric = float("-inf")
    best_epoch = -1
    best_head = -1

    action_is_verb_noun = True
    if dataset in ["COIN_anticipation"]:
        action_is_verb_noun = False

    if rank == 0:
        root_logger = logging.getLogger()
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_txt)
            for h in root_logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_txt, mode="a")
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s"
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        if action_is_verb_noun:
            csv_logger = CSVLogger(
                log_file,
                ("%d", "epoch"),
                ("%.5f", "train-acc"),
                ("%.5f", "train-acc-verb"),
                ("%.5f", "train-acc-noun"),
                ("%.5f", "train-recall"),
                ("%.5f", "train-recall-verb"),
                ("%.5f", "train-recall-noun"),
                ("%.5f", "val-acc"),
                ("%.5f", "val-acc-verb"),
                ("%.5f", "val-acc-noun"),
                ("%.5f", "val-recall"),
                ("%.5f", "val-recall-verb"),
                ("%.5f", "val-recall-noun"),
                ("%d", "val-best-head"),
                ("%d", "best-epoch"),
                ("%.5f", "best-action-recall"),
                ("%d", "best-val-head"),
                ("%d", "is-best"),
            )
        else:
            csv_logger = CSVLogger(
                log_file,
                ("%d", "epoch"),
                ("%.5f", "train-acc"),
                ("%.5f", "train-recall"),
                ("%.5f", "val-acc"),
                ("%.5f", "val-recall"),
                ("%d", "val-best-head"),
                ("%d", "best-epoch"),
                ("%.5f", "best-action-recall"),
                ("%d", "best-val-head"),
                ("%d", "is-best"),
            )

    _annotations = filter_annotations(
        dataset,
        base_path,
        train_annotations_path,
        val_annotations_path,
        file_format=file_format,
        input_format=input_format,
        video_info_path=video_info_path,
        frame_template=frame_template,
        frame_index_offset=frame_index_offset,
    )

    action_classes = _annotations["actions"]
    verb_classes = {}
    noun_classes = {}
    if action_is_verb_noun:
        verb_classes = _annotations["verbs"]
        noun_classes = _annotations["nouns"]

    val_actions = _annotations["val_actions"]
    val_verbs = {}
    val_nouns = {}
    if action_is_verb_noun:
        val_verbs = _annotations["val_verbs"]
        val_nouns = _annotations["val_nouns"]

    train_annotations = _annotations["train"]
    val_annotations = _annotations["val"]

    model = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        frames_per_second=frames_per_second,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    classifiers = init_classifier(
        embed_dim=model.embed_dim,
        num_heads=num_heads,
        verb_classes=verb_classes,
        noun_classes=noun_classes,
        action_classes=action_classes,
        num_blocks=num_probe_blocks,
        device=device,
        num_classifiers=len(opt_kwargs),
    )

    train_set, train_loader, train_data_info = init_data(
        dataset=dataset,
        training=True,
        base_path=train_data_path,
        annotations_path=train_annotations,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        fps=frames_per_second,
        anticipation_time_sec=train_anticipation_time_sec,
        anticipation_point=train_anticipation_point,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=resolution,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=False,
        input_format=input_format,
        video_info_path=video_info_path,
        frame_template=frame_template,
        frame_index_offset=frame_index_offset,
    )
    ipe = train_loader.num_batches
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    _, val_loader, _ = init_data(
        dataset=dataset,
        training=False,
        base_path=val_data_path,
        annotations_path=val_annotations,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        fps=frames_per_second,
        anticipation_time_sec=val_anticipation_time_sec,
        anticipation_point=val_anticipation_point,
        crop_size=resolution,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=False,
        input_format=input_format,
        video_info_path=video_info_path,
        frame_template=frame_template,
        frame_index_offset=frame_index_offset,
    )
    val_ipe = val_loader.num_batches
    logger.info(f"Val dataloader created... iterations per epoch: {val_ipe}")

    if ipe <= 0:
        raise ValueError(
            "train_loader.num_batches == 0. "
            "Please check base_path / input_format / video_info_path / frame_template / batch_size / world_size."
        )
    if val_ipe <= 0:
        raise ValueError(
            "val_loader.num_batches == 0. "
            "Please check validation frame folders and metadata."
        )

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    if use_fsdp_classifier and world_size > 1:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            reduce_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            buffer_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        )
        classifiers = [
            FSDP(
                c,
                device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mp_policy,
                limit_all_gathers=True,
                use_orig_params=True,
            )
            for c in classifiers
        ]
    else:
        classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]

    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
            load_opt=True,  # 先只恢复latest权重，不恢复optimizer/scaler
            use_fsdp_classifier=use_fsdp_classifier and world_size > 1,
            rank=rank,
        )

        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        if os.path.exists(best_path):
            try:
                best_ckpt = robust_checkpoint_loader(best_path, map_location=torch.device("cpu"))
                best_epoch = int(best_ckpt.get("epoch", -1))
                best_val_metrics = best_ckpt.get("val_metrics", None)
                best_head = int(best_ckpt.get("best_head", -1))
                if isinstance(best_val_metrics, dict):
                    action_metrics = best_val_metrics.get("action", {})
                    best_metric = float(action_metrics.get("recall", float("-inf")))
                    best_head = int(action_metrics.get("best_head_by_recall", best_head))
                if rank == 0:
                    logger.info(
                        f"Loaded best checkpoint metadata: epoch={best_epoch}, "
                        f"action_recall={best_metric:.5f}, head={best_head}"
                    )
            except Exception as e:
                if rank == 0:
                    logger.info(f"Could not load best checkpoint metadata: {e}")

        if val_only:
            start_epoch = 0

    def save_checkpoint(epoch, is_best=False, val_metrics=None):
        if use_fsdp_classifier and world_size > 1:
            full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            classifier_states = []
            optim_states = []

            for c, o in zip(classifiers, optimizer):
                with FSDP.state_dict_type(c, StateDictType.FULL_STATE_DICT, full_state_cfg):
                    classifier_states.append(c.state_dict())
                    # FSDP 官方推荐方式：保存 full optimizer state dict
                    optim_states.append(FSDP.full_optim_state_dict(c, o))
        else:
            classifier_states = [
                (c.module.state_dict() if isinstance(c, DistributedDataParallel) else c.state_dict())
                for c in classifiers
            ]
            optim_states = [o.state_dict() for o in optimizer]

        save_dict = {
            "classifiers": classifier_states,
            "opt": optim_states,
            "scaler": None if scaler is None else [s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
            "val_metrics": val_metrics,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "best_head": best_head,
            "use_fsdp_classifier": bool(use_fsdp_classifier and world_size > 1),
        }

        if rank == 0:
            epoch_path = os.path.join(folder, f"epoch-{epoch:03d}.pt")
            torch.save(save_dict, epoch_path)
            torch.save(save_dict, latest_path)
            if is_best:
                torch.save(save_dict, best_path)

    for epoch in range(start_epoch, num_epochs):
        if rank == 0:
            logging.info(f"Epoch {epoch}")

        train_data_info.set_epoch(epoch)

        if not val_only:
            if rank == 0:
                logging.info("Training...")
            train_metrics = train_one_epoch(
                action_is_verb_noun=action_is_verb_noun,
                ipe=ipe,
                device=device,
                model=model,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
                verb_classes=verb_classes,
                noun_classes=noun_classes,
                action_classes=action_classes,
                criterion=criterion,
                rank=rank,
            )

        val_metrics = validate(
            action_is_verb_noun=action_is_verb_noun,
            ipe=val_ipe,
            device=device,
            model=model,
            classifiers=classifiers,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            valid_verbs=val_verbs,
            valid_nouns=val_nouns,
            valid_actions=val_actions,
            verb_classes=verb_classes,
            noun_classes=noun_classes,
            action_classes=action_classes,
            criterion=criterion,
            rank=rank,
        )

        if val_only:
            if rank == 0:
                logger.info(
                    "val acc (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "val recall (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "val best head: %d "
                    % (
                        val_metrics["action"]["accuracy"],
                        val_metrics["verb"]["accuracy"],
                        val_metrics["noun"]["accuracy"],
                        val_metrics["action"]["recall"],
                        val_metrics["verb"]["recall"],
                        val_metrics["noun"]["recall"],
                        val_metrics["action"]["best_head_by_recall"],
                    )
                )
            return

        current_metric = float(val_metrics["action"]["recall"])
        val_best_head = int(val_metrics["action"]["best_head_by_recall"])
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_head = val_best_head
            if rank == 0:
                logger.info(
                    f"New best checkpoint at epoch {best_epoch} with "
                    f"val action recall {best_metric:.5f} from head {best_head}"
                )

        if action_is_verb_noun:
            if rank == 0:
                logger.info(
                    "[%5d] "
                    "train acc (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "train recall (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "val acc (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "val recall (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "val best head: %d "
                    "best head: %d "
                    % (
                        epoch + 1,
                        train_metrics["action"]["accuracy"],
                        train_metrics["verb"]["accuracy"],
                        train_metrics["noun"]["accuracy"],
                        train_metrics["action"]["recall"],
                        train_metrics["verb"]["recall"],
                        train_metrics["noun"]["recall"],
                        val_metrics["action"]["accuracy"],
                        val_metrics["verb"]["accuracy"],
                        val_metrics["noun"]["accuracy"],
                        val_metrics["action"]["recall"],
                        val_metrics["verb"]["recall"],
                        val_metrics["noun"]["recall"],
                        val_best_head,
                        best_head,
                    )
                )
                csv_logger.log(
                    epoch + 1,
                    train_metrics["action"]["accuracy"],
                    train_metrics["verb"]["accuracy"],
                    train_metrics["noun"]["accuracy"],
                    train_metrics["action"]["recall"],
                    train_metrics["verb"]["recall"],
                    train_metrics["noun"]["recall"],
                    val_metrics["action"]["accuracy"],
                    val_metrics["verb"]["accuracy"],
                    val_metrics["noun"]["accuracy"],
                    val_metrics["action"]["recall"],
                    val_metrics["verb"]["recall"],
                    val_metrics["noun"]["recall"],
                    val_best_head,
                    best_epoch,
                    best_metric,
                    best_head,
                    int(is_best),
                )
        else:
            if rank == 0:
                logger.info(
                    "[%5d] "
                    "train acc (v/n): %.1f%% "
                    "train recall (v/n): %.1f%% "
                    "val acc (v/n): %.1f%% "
                    "val recall (v/n): %.1f%% "
                    "val best head: %d "
                    "best head: %d "
                    % (
                        epoch + 1,
                        train_metrics["action"]["accuracy"],
                        train_metrics["action"]["recall"],
                        val_metrics["action"]["accuracy"],
                        val_metrics["action"]["recall"],
                        val_best_head,
                        best_head,
                    )
                )
                csv_logger.log(
                    epoch + 1,
                    train_metrics["action"]["accuracy"],
                    train_metrics["action"]["recall"],
                    val_metrics["action"]["accuracy"],
                    val_metrics["action"]["recall"],
                    val_best_head,
                    best_epoch,
                    best_metric,
                    best_head,
                    int(is_best),
                )

        save_checkpoint(epoch + 1, is_best=is_best, val_metrics=val_metrics)


def train_one_epoch(
    action_is_verb_noun,
    ipe,
    device,
    model,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    noun_classes,
    verb_classes,
    action_classes,
    criterion,
    rank=0,
):
    if ipe <= 0:
        raise ValueError("train_one_epoch received ipe <= 0.")

    _data_loader = iter(data_loader)
    for c in classifiers:
        c.train(mode=True)

    if action_is_verb_noun:
        verb_metric_loggers = [ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers]
        noun_metric_loggers = [ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers]
    action_metric_loggers = [ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers]
    data_elapsed_time_meter = AverageMeter()

    for itr in range(ipe):
        itr_start_time = time.time()

        try:
            udata = next(_data_loader)
        except Exception:
            _data_loader = iter(data_loader)
            udata = next(_data_loader)

        [s.step() for s in scheduler]
        [wds.step() for wds in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            clips = udata[0].to(device)
            anticipation_times = udata[-1].to(device)

            if action_is_verb_noun:
                _verbs, _nouns = udata[1], udata[2]
                verb_labels, noun_labels, action_labels = [], [], []
                for v, n in zip(_verbs, _nouns):
                    verb_labels.append(verb_classes[int(v)])
                    noun_labels.append(noun_classes[int(n)])
                    action_labels.append(action_classes[(int(v), int(n))])
                verb_labels = torch.tensor(verb_labels).to(device).to(_verbs.dtype)
                noun_labels = torch.tensor(noun_labels).to(device).to(_verbs.dtype)
                action_labels = torch.tensor(action_labels).to(device).to(_verbs.dtype)
            else:
                _actions = udata[1]
                action_labels = [action_classes[str(int(a))] for a in _actions]
                action_labels = torch.tensor(action_labels).to(device).to(_actions.dtype)

            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            with torch.no_grad():
                outputs = model(clips, anticipation_times)
            outputs = [c(outputs) for c in classifiers]

        if action_is_verb_noun:
            verb_loss = [criterion(o["verb"], verb_labels) for o in outputs]
            noun_loss = [criterion(o["noun"], noun_labels) for o in outputs]
            action_loss = [criterion(o["action"], action_labels) for o in outputs]
            loss = [v + n + a for v, n, a in zip(verb_loss, noun_loss, action_loss)]
        else:
            loss = [criterion(o["action"], action_labels) for o in outputs]

        if scaler is not None:
            [s.scale(l).backward() for s, l in zip(scaler, loss)]
            [s.step(o) for s, o in zip(scaler, optimizer)]
            [s.update() for s in scaler]
        else:
            [L.backward() for L in loss]
            [o.step() for o in optimizer]

        [o.zero_grad(set_to_none=True) for o in optimizer]

        with torch.no_grad():
            if action_is_verb_noun:
                verb_metrics = [m(o["verb"], verb_labels) for o, m in zip(outputs, verb_metric_loggers)]
                noun_metrics = [m(o["noun"], noun_labels) for o, m in zip(outputs, noun_metric_loggers)]
            action_metrics = [m(o["action"], action_labels) for o, m in zip(outputs, action_metric_loggers)]

        if rank == 0 and (itr % 10 == 0 or itr == ipe - 1):
            if action_is_verb_noun:
                logger.info(
                    "[%5d] "
                    "acc (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "recall (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "[mem: %.2e] "
                    "[data: %.1f ms]"
                    % (
                        itr,
                        max([a["accuracy"] for a in action_metrics]),
                        max([v["accuracy"] for v in verb_metrics]),
                        max([n["accuracy"] for n in noun_metrics]),
                        max([a["recall"] for a in action_metrics]),
                        max([v["recall"] for v in verb_metrics]),
                        max([n["recall"] for n in noun_metrics]),
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                        data_elapsed_time_meter.avg,
                    )
                )
            else:
                logger.info(
                    "[%5d] "
                    "acc (v/n): %.1f%% "
                    "recall (v/n): %.1f%% "
                    "[mem: %.2e] "
                    "[data: %.1f ms]"
                    % (
                        itr,
                        max([a["accuracy"] for a in action_metrics]),
                        max([a["recall"] for a in action_metrics]),
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                        data_elapsed_time_meter.avg,
                    )
                )

    del _data_loader
    ret = dict(
        action=_summarize_probe_head_metrics(action_metrics),
    )
    if action_is_verb_noun:
        ret.update(
            dict(
                verb=_summarize_probe_head_metrics(verb_metrics),
                noun=_summarize_probe_head_metrics(noun_metrics),
            )
        )
    return ret


@torch.no_grad()
def validate(
    action_is_verb_noun,
    ipe,
    device,
    model,
    classifiers,
    data_loader,
    use_bfloat16,
    valid_nouns,
    valid_verbs,
    valid_actions,
    noun_classes,
    verb_classes,
    action_classes,
    criterion,
    rank=0,
):
    del valid_nouns, valid_verbs, valid_actions  # kept for compatibility

    if ipe <= 0:
        raise ValueError("validate received ipe <= 0.")

    if rank == 0:
        logger.info("Running val...")
    _data_loader = iter(data_loader)
    for c in classifiers:
        c.train(mode=False)

    if action_is_verb_noun:
        verb_metric_loggers = [ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers]
        noun_metric_loggers = [ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers]
    action_metric_loggers = [ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers]

    for itr in range(ipe):
        try:
            udata = next(_data_loader)
        except Exception:
            _data_loader = iter(data_loader)
            udata = next(_data_loader)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            clips = udata[0].to(device)
            anticipation_times = udata[-1].to(device)

            if action_is_verb_noun:
                _verbs, _nouns = udata[1], udata[2]
                verb_labels, noun_labels, action_labels = [], [], []
                for v, n in zip(_verbs, _nouns):
                    verb_labels.append(verb_classes[int(v)])
                    noun_labels.append(noun_classes[int(n)])
                    action_labels.append(action_classes[(int(v), int(n))])
                verb_labels = torch.tensor(verb_labels).to(device).to(_verbs.dtype)
                noun_labels = torch.tensor(noun_labels).to(device).to(_verbs.dtype)
                action_labels = torch.tensor(action_labels).to(device).to(_verbs.dtype)
            else:
                _actions = udata[1]
                action_labels = [action_classes[str(int(a))] for a in _actions]
                action_labels = torch.tensor(action_labels).to(device).to(_actions.dtype)

            outputs = model(clips, anticipation_times)
            outputs = [c(outputs) for c in classifiers]

            if action_is_verb_noun:
                verb_loss = sum([criterion(o["verb"], verb_labels) for o in outputs])
                noun_loss = sum([criterion(o["noun"], noun_labels) for o in outputs])
                action_loss = sum([criterion(o["action"], action_labels) for o in outputs])
                loss = verb_loss + noun_loss + action_loss
            else:
                loss = sum([criterion(o["action"], action_labels) for o in outputs])

            action_metrics = [m(o["action"], action_labels) for o, m in zip(outputs, action_metric_loggers)]
            if action_is_verb_noun:
                verb_metrics = [m(o["verb"], verb_labels) for o, m in zip(outputs, verb_metric_loggers)]
                noun_metrics = [m(o["noun"], noun_labels) for o, m in zip(outputs, noun_metric_loggers)]

        if rank == 0 and (itr % 10 == 0 or itr == ipe - 1):
            if action_is_verb_noun:
                logger.info(
                    "[%5d] "
                    "acc (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "recall (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "loss (v/n): %.3f (%.3f %.3f) "
                    "[mem: %.2e] "
                    % (
                        itr,
                        max([a["accuracy"] for a in action_metrics]),
                        max([v["accuracy"] for v in verb_metrics]),
                        max([n["accuracy"] for n in noun_metrics]),
                        max([a["recall"] for a in action_metrics]),
                        max([v["recall"] for v in verb_metrics]),
                        max([n["recall"] for n in noun_metrics]),
                        loss,
                        verb_loss,
                        noun_loss,
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                    )
                )
            else:
                logger.info(
                    "[%5d] "
                    "acc (v/n): %.1f%% "
                    "recall (v/n): %.1f%% "
                    "loss (v/n): %.3f "
                    "[mem: %.2e] "
                    % (
                        itr,
                        max([a["accuracy"] for a in action_metrics]),
                        max([a["recall"] for a in action_metrics]),
                        loss,
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                    )
                )

    del _data_loader
    ret = dict(
        action=_summarize_probe_head_metrics(action_metrics),
    )
    if action_is_verb_noun:
        ret.update(
            dict(
                verb=_summarize_probe_head_metrics(verb_metrics),
                noun=_summarize_probe_head_metrics(noun_metrics),
            )
        )
    return ret


def load_checkpoint(
    device,
    r_path,
    classifiers,
    opt,
    scaler,
    val_only=False,
    load_opt=True,
    use_fsdp_classifier=False,
    rank=0,
):
    del device  # kept for compatibility

    logger.info(f"read-path: {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    if "classifiers" not in checkpoint:
        raise ValueError(f"Checkpoint at {r_path} does not contain key: 'classifiers'")

    msg = []
    for c, pd in zip(classifiers, checkpoint["classifiers"]):
        if isinstance(c, DistributedDataParallel):
            msg.append(c.module.load_state_dict(pd))
        elif isinstance(c, FSDP):
            full_state_cfg = FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
            with FSDP.state_dict_type(c, StateDictType.FULL_STATE_DICT, full_state_cfg):
                msg.append(c.load_state_dict(pd))
        else:
            msg.append(c.load_state_dict(pd))

    epoch = int(checkpoint.get("epoch", 0))
    logger.info(f"loaded classifier weights from epoch {epoch} with msg: {msg}")

    if val_only:
        logger.info("val_only=True, skip loading optimizer/scaler states")
        return classifiers, opt, scaler, 0

    if not load_opt:
        logger.info("load_opt=False, skip loading optimizer/scaler states")
        return classifiers, opt, scaler, epoch

    if "opt" not in checkpoint or checkpoint["opt"] is None:
        raise ValueError(f"Checkpoint at {r_path} does not contain optimizer state.")

    if len(checkpoint["opt"]) != len(opt):
        raise ValueError(
            f"Number of optimizer states mismatch: "
            f"saved={len(checkpoint['opt'])}, current={len(opt)}"
        )

    for i, (c, o, saved_opt) in enumerate(zip(classifiers, opt, checkpoint["opt"])):
        if isinstance(c, FSDP) and use_fsdp_classifier:
            logger.info(f"[rank {rank}] loading FSDP optimizer state for optimizer #{i}")
            loaded_osd = FSDP.optim_state_dict_to_load(
                model=c,
                optim=o,
                optim_state_dict=saved_opt,
            )
            o.load_state_dict(loaded_osd)
        else:
            logger.info(f"[rank {rank}] loading regular optimizer state for optimizer #{i}")
            o.load_state_dict(saved_opt)

    if scaler is not None and checkpoint.get("scaler", None) is not None:
        if len(checkpoint["scaler"]) != len(scaler):
            raise ValueError(
                f"Number of scaler states mismatch: "
                f"saved={len(checkpoint['scaler'])}, current={len(scaler)}"
            )
        for i, (s, sd) in enumerate(zip(scaler, checkpoint["scaler"])):
            logger.info(f"[rank {rank}] loading scaler state for scaler #{i}")
            s.load_state_dict(sd)
    else:
        logger.info("No scaler state found in checkpoint, or scaler is None")

    logger.info(f"loaded optimizers/scalers from epoch {epoch}")
    return classifiers, opt, scaler, epoch
