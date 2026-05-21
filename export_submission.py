#!/usr/bin/env python3
"""Export an EPIC-KITCHENS-100 action anticipation submission from a frozen probe."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from evals.action_anticipation_frozen.dataloader import make_transforms
from evals.action_anticipation_frozen.models import init_classifier, init_module


LOGGER = logging.getLogger("export_ek100_submission")
MISSING_CLASS_LOGIT = -1.0e9


def build_class_maps(train_csv: str) -> Tuple[Dict[int, int], Dict[int, int], Dict[Tuple[int, int], int]]:
    df = pd.read_csv(train_csv)
    actions = set(zip(df["verb_class"].astype(int).tolist(), df["noun_class"].astype(int).tolist()))
    verbs = set(v for v, _ in actions)
    nouns = set(n for _, n in actions)
    return (
        {k: i for i, k in enumerate(verbs)},
        {k: i for i, k in enumerate(nouns)},
        {k: i for i, k in enumerate(actions)},
    )


def invert_map(d: Dict) -> Dict:
    return {v: k for k, v in d.items()}


def count_nonfinite_tensors(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    nan_count = 0
    inf_count = 0
    for value in state_dict.values():
        if torch.is_tensor(value) and value.is_floating_point():
            nan_count += int(torch.isnan(value).sum().item())
            inf_count += int(torch.isinf(value).sum().item())
    return nan_count, inf_count


def resolve_video_dir(base_path: str, video_id: str) -> str:
    participant_id = video_id.split("_")[0]
    candidates = [
        os.path.join(base_path, video_id),
        os.path.join(base_path, participant_id, video_id),
        os.path.join(base_path, participant_id, "rgb_frames", video_id),
        os.path.join(base_path, "rgb_frames", video_id),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Frame directory not found for {video_id} under {base_path}")


def read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


class EK100TestDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        test_csv: str,
        video_info_csv: str,
        frames_per_clip: int,
        fps: int,
        resolution: int,
        anticipation_time_sec: float,
        anticipation_point: float,
        frame_template: str,
        frame_index_offset: int,
    ) -> None:
        self.base_path = base_path
        self.df = pd.read_csv(test_csv)
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
        narration_id = str(row["narration_id"])
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
            start_frame * self.anticipation_point + (1.0 - self.anticipation_point) * stop_frame - anticipation_frames
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
        anticipation_time = torch.tensor(float(self.anticipation_time_sec), dtype=torch.float32)
        return narration_id, clip, anticipation_time


def load_yaml(path: str) -> dict:
    with open(path, "r") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def build_backbone(config: dict, device: torch.device) -> torch.nn.Module:
    data_cfg = config["experiment"]["data"]
    model_cfg = config["model_kwargs"]
    return init_module(
        module_name=model_cfg["module_name"],
        device=device,
        frames_per_clip=int(data_cfg["frames_per_clip"]),
        frames_per_second=int(data_cfg["frames_per_second"]),
        resolution=int(data_cfg["resolution"]),
        checkpoint=model_cfg["checkpoint"],
        model_kwargs=model_cfg["pretrain_kwargs"],
        wrapper_kwargs=model_cfg["wrapper_kwargs"],
    )


def load_classifiers(
    checkpoint_path: str,
    embed_dim: int,
    device: torch.device,
    config: dict,
    verb_classes: Dict[int, int],
    noun_classes: Dict[int, int],
    action_classes: Dict[Tuple[int, int], int],
) -> Tuple[List[torch.nn.Module], List[int], Optional[int]]:
    LOGGER.info("Loading probe checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    checkpoint_best_head = checkpoint.get("best_head")
    if checkpoint_best_head is not None:
        checkpoint_best_head = int(checkpoint_best_head)
    all_classifier_states = checkpoint["classifiers"]
    selected_states = []
    skipped = []
    for index, state_dict in enumerate(all_classifier_states):
        nan_count, inf_count = count_nonfinite_tensors(state_dict)
        if nan_count or inf_count:
            skipped.append((index, nan_count, inf_count))
            continue
        selected_states.append((index, state_dict))

    if not selected_states:
        raise RuntimeError(f"No finite classifier heads found in {checkpoint_path}")

    if skipped:
        LOGGER.warning("Skipping non-finite probe heads: %s", skipped)
    selected_indices = [index for index, _ in selected_states]
    LOGGER.info("Using finite probe head indices: %s", selected_indices)

    classifier_states = [state_dict for _, state_dict in selected_states]
    classifier_cfg = config["experiment"]["classifier"]
    classifiers = init_classifier(
        embed_dim=embed_dim,
        num_heads=int(classifier_cfg["num_heads"]),
        num_blocks=int(classifier_cfg["num_probe_blocks"]),
        device=device,
        num_classifiers=len(classifier_states),
        action_classes=action_classes,
        verb_classes=verb_classes,
        noun_classes=noun_classes,
    )
    for classifier, state_dict in zip(classifiers, classifier_states):
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        classifier.load_state_dict(cleaned, strict=True)
        classifier.eval()
    LOGGER.info(
        "Loaded %d finite probe heads from epoch %s; checkpoint best_head=%s",
        len(classifiers),
        checkpoint.get("epoch", "unknown"),
        checkpoint_best_head,
    )
    return classifiers, selected_indices, checkpoint_best_head


def expand_scores(logits: torch.Tensor, inverse_map: Dict[int, int], total_classes: int) -> Dict[str, float]:
    values = {str(i): float(MISSING_CLASS_LOGIT) for i in range(total_classes)}
    for head_index, original_class in inverse_map.items():
        values[str(int(original_class))] = float(logits[int(head_index)].item())
    return values


def top_action_scores(logits: torch.Tensor, inverse_map: Dict[int, Tuple[int, int]], k: int = 100) -> Dict[str, float]:
    top = torch.topk(logits, k=min(k, logits.numel()), dim=-1)
    scores = {}
    for head_index, score in zip(top.indices.tolist(), top.values.tolist()):
        verb_id, noun_id = inverse_map[int(head_index)]
        scores[f"{int(verb_id)},{int(noun_id)}"] = float(score)
    return scores


def run_export(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    config = load_yaml(args.config)
    data_cfg = config["experiment"]["data"]
    if args.backbone_checkpoint:
        config["model_kwargs"]["checkpoint"] = args.backbone_checkpoint

    train_csv = args.train_csv or data_cfg["dataset_train"]
    test_csv = args.test_csv or os.path.join(os.path.dirname(data_cfg["dataset_val"]), "EPIC_100_test_timestamps.csv")
    video_info_csv = args.video_info_csv or data_cfg["video_info_path"]
    base_path = args.base_path or data_cfg["base_path"]
    anticipation_range = data_cfg.get("anticipation_time_sec", [1.0, 1.0])
    anticipation_time_sec = float(args.anticipation_time_sec if args.anticipation_time_sec is not None else anticipation_range[0])
    anticipation_point_range = data_cfg.get("val_anticipation_point", [1.0, 1.0])
    anticipation_point = float(args.anticipation_point if args.anticipation_point is not None else anticipation_point_range[0])

    output_json = Path(args.output_json).resolve()
    output_zip = Path(args.output_zip or output_json.with_suffix(".zip")).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    LOGGER.info("Using device: %s", device)

    verb_classes, noun_classes, action_classes = build_class_maps(train_csv)
    inverse_verb = invert_map(verb_classes)
    inverse_noun = invert_map(noun_classes)
    inverse_action = invert_map(action_classes)
    LOGGER.info(
        "Class maps: verbs=%d nouns=%d train_actions=%d",
        len(verb_classes),
        len(noun_classes),
        len(action_classes),
    )

    dataset = EK100TestDataset(
        base_path=base_path,
        test_csv=test_csv,
        video_info_csv=video_info_csv,
        frames_per_clip=int(data_cfg["frames_per_clip"]),
        fps=int(data_cfg["frames_per_second"]),
        resolution=int(data_cfg["resolution"]),
        anticipation_time_sec=anticipation_time_sec,
        anticipation_point=anticipation_point,
        frame_template=data_cfg.get("frame_template", "frame_{:010d}.jpg"),
        frame_index_offset=int(data_cfg.get("frame_index_offset", 1)),
    )
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(int(args.max_samples), len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    LOGGER.info("Test samples: %d", len(dataset))

    if args.verbose_model:
        backbone = build_backbone(config, device)
        classifiers, classifier_indices, checkpoint_best_head = load_classifiers(
            checkpoint_path=args.probe_checkpoint,
            embed_dim=backbone.embed_dim,
            device=device,
            config=config,
            verb_classes=verb_classes,
            noun_classes=noun_classes,
            action_classes=action_classes,
        )
    else:
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            backbone = build_backbone(config, device)
            classifiers, classifier_indices, checkpoint_best_head = load_classifiers(
                checkpoint_path=args.probe_checkpoint,
                embed_dim=backbone.embed_dim,
                device=device,
                config=config,
                verb_classes=verb_classes,
                noun_classes=noun_classes,
                action_classes=action_classes,
            )
    if args.classifier_index >= 0:
        if args.classifier_index not in classifier_indices:
            raise ValueError(
                f"Requested original probe head {args.classifier_index}, "
                f"but available finite heads are {classifier_indices}"
            )
        classifiers = [classifiers[classifier_indices.index(args.classifier_index)]]
        classifier_indices = [args.classifier_index]
        LOGGER.info("Using original probe head index %d", args.classifier_index)
    elif args.average_heads:
        LOGGER.warning(
            "Averaging logits across finite probe heads %s. "
            "This is an explicit override; the report protocol exports the validation-selected best head.",
            classifier_indices,
        )
    elif checkpoint_best_head is not None and checkpoint_best_head in classifier_indices:
        classifiers = [classifiers[classifier_indices.index(checkpoint_best_head)]]
        classifier_indices = [checkpoint_best_head]
        LOGGER.info("Using checkpoint best_head %d, matching the report export protocol", checkpoint_best_head)
    elif len(classifier_indices) == 1:
        LOGGER.info("Only one finite probe head is available; using head %d", classifier_indices[0])
    else:
        raise ValueError(
            "Probe checkpoint does not contain a finite best_head. "
            "Pass --classifier-index to export the validation-selected head, "
            "or pass --average-heads to explicitly average all finite heads."
        )

    payload = {
        "version": "0.2",
        "challenge": "action_anticipation",
        "sls_pt": args.sls_pt,
        "sls_tl": args.sls_tl,
        "sls_td": args.sls_td,
        "results": {},
    }

    start_time = time.time()
    use_autocast = device.type == "cuda" and args.use_bfloat16
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            narration_ids, clips, anticipation_times = batch
            clips = clips.to(device, non_blocking=True)
            anticipation_times = anticipation_times.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_autocast):
                features = backbone(clips, anticipation_times)
                outputs = [classifier(features) for classifier in classifiers]

            verb_logits = torch.stack([out["verb"].detach().float().cpu() for out in outputs], dim=0).mean(dim=0)
            noun_logits = torch.stack([out["noun"].detach().float().cpu() for out in outputs], dim=0).mean(dim=0)
            action_logits = torch.stack([out["action"].detach().float().cpu() for out in outputs], dim=0).mean(dim=0)
            if not torch.isfinite(verb_logits).all():
                raise RuntimeError(f"Non-finite verb logits at dataloader step {step}")
            if not torch.isfinite(noun_logits).all():
                raise RuntimeError(f"Non-finite noun logits at dataloader step {step}")
            if not args.omit_action and not torch.isfinite(action_logits).all():
                raise RuntimeError(f"Non-finite action logits at dataloader step {step}")

            for i, narration_id in enumerate(narration_ids):
                entry = {
                    "verb": expand_scores(verb_logits[i], inverse_verb, args.num_verb_classes),
                    "noun": expand_scores(noun_logits[i], inverse_noun, args.num_noun_classes),
                }
                if not args.omit_action:
                    entry["action"] = top_action_scores(action_logits[i], inverse_action, k=100)
                payload["results"][str(narration_id)] = entry

            if step % args.log_every == 0 or step == len(loader):
                done = min(step * args.batch_size, len(dataset))
                elapsed = time.time() - start_time
                rate = done / max(elapsed, 1.0)
                remaining = (len(dataset) - done) / max(rate, 1.0e-9)
                LOGGER.info(
                    "Processed %d/%d samples (%.2f samples/s, ETA %.1f min)",
                    done,
                    len(dataset),
                    rate,
                    remaining / 60.0,
                )

    if len(payload["results"]) != len(dataset):
        raise RuntimeError(f"Expected {len(dataset)} results, wrote {len(payload['results'])}")

    with open(output_json, "w") as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(output_json, arcname="test.json")
    LOGGER.info("Wrote %s", output_json)
    LOGGER.info("Wrote flat archive %s", output_zip)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jfaa_vjepa2_vitG_infer.yaml")
    parser.add_argument("--probe-checkpoint", required=True)
    parser.add_argument("--backbone-checkpoint")
    parser.add_argument("--train-csv")
    parser.add_argument("--test-csv")
    parser.add_argument("--video-info-csv")
    parser.add_argument("--base-path")
    parser.add_argument("--output-json", default="outputs/submissions/epoch-009/test.json")
    parser.add_argument("--output-zip")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verbose-model", action="store_true")
    parser.add_argument("--classifier-index", type=int, default=-1)
    parser.add_argument(
        "--average-heads",
        action="store_true",
        help="Explicitly average all finite probe heads instead of using checkpoint best_head.",
    )
    parser.add_argument("--anticipation-time-sec", type=float)
    parser.add_argument("--anticipation-point", type=float)
    parser.add_argument("--num-verb-classes", type=int, default=97)
    parser.add_argument("--num-noun-classes", type=int, default=300)
    parser.add_argument("--sls-pt", type=int, default=2)
    parser.add_argument("--sls-tl", type=int, default=3)
    parser.add_argument("--sls-td", type=int, default=3)
    parser.add_argument("--omit-action", action="store_true")
    parser.add_argument("--no-bfloat16", dest="use_bfloat16", action="store_false")
    parser.set_defaults(use_bfloat16=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_export(parse_args())
