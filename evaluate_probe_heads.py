#!/usr/bin/env python3
"""Evaluate EK100 frozen probe heads with the same single-head validation paradigm."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from evals.action_anticipation_frozen.dataloader import make_transforms
from export_submission import (
    build_backbone,
    build_class_maps,
    load_classifiers,
    load_yaml,
    read_rgb,
    resolve_video_dir,
)


LOGGER = logging.getLogger("evaluate_ek100_probe_heads")


class EK100ValidationDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        val_csv: str,
        video_info_csv: str,
        verb_classes: Dict[int, int],
        noun_classes: Dict[int, int],
        action_classes: Dict[Tuple[int, int], int],
        frames_per_clip: int,
        fps: int,
        resolution: int,
        anticipation_time_sec: float,
        anticipation_point: float,
        frame_template: str,
        frame_index_offset: int,
    ) -> None:
        df = pd.read_csv(val_csv)
        keep = [
            (int(v), int(n)) in action_classes
            for v, n in zip(df["verb_class"].tolist(), df["noun_class"].tolist())
        ]
        self.df = df[keep].reset_index(drop=True)
        self.base_path = base_path
        self.verb_classes = verb_classes
        self.noun_classes = noun_classes
        self.action_classes = action_classes
        self.frames_per_clip = frames_per_clip
        self.target_fps = fps
        self.anticipation_time_sec = anticipation_time_sec
        self.anticipation_point = anticipation_point
        self.frame_template = frame_template
        self.frame_index_offset = frame_index_offset
        self.transform = make_transforms(training=False, crop_size=resolution)

        info_df = pd.read_csv(video_info_csv)
        self.fps_lookup = {r["video_id"]: float(r["fps"]) for _, r in info_df.iterrows()}
        self.num_frames_lookup = {}
        if "num_frames" in info_df.columns:
            self.num_frames_lookup = {r["video_id"]: int(r["num_frames"]) for _, r in info_df.iterrows()}
        self.video_dir_lookup = {
            video_id: resolve_video_dir(base_path, video_id)
            for video_id in self.df["video_id"].drop_duplicates().tolist()
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        video_id = str(row["video_id"])
        video_dir = self.video_dir_lookup[video_id]
        video_fps = self.fps_lookup[video_id]
        max_frame_index = self.num_frames_lookup.get(video_id, None)
        if max_frame_index is None:
            max_frame_index = len([p for p in os.scandir(video_dir) if p.is_file()])
        max_frame_index = int(max_frame_index) - 1

        frame_stride = max(1, int(round(video_fps / self.target_fps)))
        num_input_frames = int(self.frames_per_clip * frame_stride)
        anticipation_frames = int(round(self.anticipation_time_sec * video_fps))
        start_frame = int(row["start_frame"])
        stop_frame = int(row["stop_frame"])
        anticipation_frame = int(
            start_frame * self.anticipation_point
            + (1.0 - self.anticipation_point) * stop_frame
            - anticipation_frames
        )

        indices = np.arange(anticipation_frame - num_input_frames, anticipation_frame, frame_stride).astype(np.int64)
        indices[indices < 0] = 0
        indices[indices > max_frame_index] = max_frame_index
        frame_paths = [
            os.path.join(video_dir, self.frame_template.format(int(idx) + self.frame_index_offset))
            for idx in indices
        ]

        buffer = np.stack([read_rgb(path) for path in frame_paths], axis=0)
        clip = self.transform(buffer)
        verb_id = int(row["verb_class"])
        noun_id = int(row["noun_class"])
        action_key = (verb_id, noun_id)
        return (
            str(row["narration_id"]),
            clip,
            torch.tensor(float(self.anticipation_time_sec), dtype=torch.float32),
            torch.tensor(self.verb_classes[verb_id], dtype=torch.long),
            torch.tensor(self.noun_classes[noun_id], dtype=torch.long),
            torch.tensor(self.action_classes[action_key], dtype=torch.long),
        )


class TopKStats:
    def __init__(self, num_classes: int, k: int) -> None:
        self.tp = torch.zeros(num_classes, dtype=torch.long)
        self.fn = torch.zeros(num_classes, dtype=torch.long)
        self.top1 = 0
        self.total = 0
        self.k = k

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        logits = logits.detach().float().cpu()
        labels = labels.detach().long().cpu()
        topk = logits.topk(min(self.k, logits.shape[1]), dim=1).indices
        top1 = topk[:, 0]
        self.top1 += int((top1 == labels).sum().item())
        self.total += int(labels.numel())
        for pred, label in zip(topk, labels):
            if int(label) in pred.tolist():
                self.tp[int(label)] += 1
            else:
                self.fn[int(label)] += 1

    def compute(self) -> dict:
        seen = (self.tp + self.fn) > 0
        recall = float((100.0 * (self.tp[seen].float() / (self.tp[seen] + self.fn[seen]).float())).mean().item())
        topk_acc = float(100.0 * self.tp.sum().item() / max(1, int((self.tp + self.fn).sum().item())))
        top1_acc = float(100.0 * self.top1 / max(1, self.total))
        return {
            "top1_accuracy": top1_acc,
            "top5_accuracy": topk_acc,
            "top5_mean_recall": recall,
            "observed_classes": int(seen.sum().item()),
            "samples": int(self.total),
        }


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    config = load_yaml(args.config)
    data_cfg = config["experiment"]["data"]
    if args.backbone_checkpoint:
        config["model_kwargs"]["checkpoint"] = args.backbone_checkpoint

    train_csv = args.train_csv or data_cfg["dataset_train"]
    val_csv = args.val_csv or data_cfg["dataset_val"]
    base_path = args.base_path or data_cfg["base_path"]
    video_info_csv = args.video_info_csv or data_cfg["video_info_path"]
    anticipation_range = data_cfg.get("anticipation_time_sec", [1.0, 1.0])
    anticipation_time_sec = float(args.anticipation_time_sec if args.anticipation_time_sec is not None else anticipation_range[0])
    anticipation_point_range = data_cfg.get("val_anticipation_point", [1.0, 1.0])
    anticipation_point = float(args.anticipation_point if args.anticipation_point is not None else anticipation_point_range[0])

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    verb_classes, noun_classes, action_classes = build_class_maps(train_csv)
    dataset = EK100ValidationDataset(
        base_path=base_path,
        val_csv=val_csv,
        video_info_csv=video_info_csv,
        verb_classes=verb_classes,
        noun_classes=noun_classes,
        action_classes=action_classes,
        frames_per_clip=int(data_cfg["frames_per_clip"]),
        fps=int(data_cfg["frames_per_second"]),
        resolution=int(data_cfg["resolution"]),
        anticipation_time_sec=anticipation_time_sec,
        anticipation_point=anticipation_point,
        frame_template=data_cfg.get("frame_template", "frame_{:010d}.jpg"),
        frame_index_offset=int(data_cfg.get("frame_index_offset", 1)),
    )
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    LOGGER.info("Validation samples: %d", len(dataset))

    if args.verbose_model:
        backbone = build_backbone(config, device)
        classifiers, classifier_indices, _ = load_classifiers(
            args.probe_checkpoint,
            backbone.embed_dim,
            device,
            config,
            verb_classes,
            noun_classes,
            action_classes,
        )
    else:
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            backbone = build_backbone(config, device)
            classifiers, classifier_indices, _ = load_classifiers(
                args.probe_checkpoint,
                backbone.embed_dim,
                device,
                config,
                verb_classes,
                noun_classes,
                action_classes,
            )

    selected = None
    if args.classifier_index >= 0:
        if args.classifier_index not in classifier_indices:
            raise ValueError(f"Requested head {args.classifier_index}; finite heads are {classifier_indices}")
        selected = classifier_indices.index(args.classifier_index)
        classifiers = [classifiers[selected]]
        classifier_indices = [args.classifier_index]
    LOGGER.info("Evaluating original probe head indices: %s", classifier_indices)

    stats = {
        head: {
            "action": TopKStats(len(action_classes), k=5),
            "verb": TopKStats(len(verb_classes), k=5),
            "noun": TopKStats(len(noun_classes), k=5),
        }
        for head in classifier_indices
    }

    use_autocast = device.type == "cuda" and args.use_bfloat16
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            _, clips, anticipation_times, verb_labels, noun_labels, action_labels = batch
            clips = clips.to(device, non_blocking=True)
            anticipation_times = anticipation_times.to(device, non_blocking=True)
            verb_labels = verb_labels.to(device, non_blocking=True)
            noun_labels = noun_labels.to(device, non_blocking=True)
            action_labels = action_labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_autocast):
                features = backbone(clips, anticipation_times)
                outputs = [classifier(features) for classifier in classifiers]

            for head, output in zip(classifier_indices, outputs):
                for key in ("action", "verb", "noun"):
                    if not torch.isfinite(output[key]).all():
                        raise RuntimeError(f"Non-finite {key} logits for head {head} at step {step}")
                stats[head]["action"].update(output["action"], action_labels)
                stats[head]["verb"].update(output["verb"], verb_labels)
                stats[head]["noun"].update(output["noun"], noun_labels)

            if step % args.log_every == 0 or step == len(loader):
                LOGGER.info("Processed %d/%d batches", step, len(loader))

    metrics = {
        str(head): {
            "action": stats[head]["action"].compute(),
            "verb": stats[head]["verb"].compute(),
            "noun": stats[head]["noun"].compute(),
        }
        for head in classifier_indices
    }
    best_head = max(
        classifier_indices,
        key=lambda head: metrics[str(head)]["action"]["top5_mean_recall"],
    )
    payload = {
        "best_head_by_action_top5_mean_recall": int(best_head),
        "metrics": metrics,
        "config": {
            "samples": len(dataset),
            "anticipation_time_sec": anticipation_time_sec,
            "anticipation_point": anticipation_point,
            "finite_heads": classifier_indices,
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    LOGGER.info("Best head by action top5 mean recall: %s", best_head)
    LOGGER.info("Wrote %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jfaa_vjepa2_vitG_infer.yaml")
    parser.add_argument("--probe-checkpoint", required=True)
    parser.add_argument("--backbone-checkpoint")
    parser.add_argument("--train-csv")
    parser.add_argument("--val-csv")
    parser.add_argument("--video-info-csv")
    parser.add_argument("--base-path")
    parser.add_argument("--output-json", default="outputs/submissions/epoch-009/val_head_metrics.json")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--classifier-index", type=int, default=-1)
    parser.add_argument("--anticipation-time-sec", type=float)
    parser.add_argument("--anticipation-point", type=float)
    parser.add_argument("--verbose-model", action="store_true")
    parser.add_argument("--no-bfloat16", dest="use_bfloat16", action="store_false")
    parser.set_defaults(use_bfloat16=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
