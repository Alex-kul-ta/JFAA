# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import multiprocessing
import os
import random
from dataclasses import dataclass
from itertools import islice
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from src.datasets.utils.worker_init_fn import pl_worker_init_function

multiprocessing.set_start_method("spawn", force=True)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards_list):
    num_shards = len(shards_list)
    total_size = num_shards
    return total_size, num_shards


class split_by_node(wds.PipelineStage):
    """Node splitter that uses provided rank/world_size instead of torch.distributed state."""

    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size

    def run(self, src):
        if self.world_size > 1:
            yield from islice(src, self.rank, None, self.world_size)
        else:
            yield from src


def _sample_scalar_or_range(x):
    if isinstance(x, (list, tuple)):
        assert len(x) == 2, f"Expected range with len=2, got {x}"
        return random.uniform(float(x[0]), float(x[1]))
    return float(x)


def _read_rgb(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _count_image_files(video_dir):
    count = 0
    for entry in os.scandir(video_dir):
        if entry.is_file() and entry.name.lower().endswith(_IMAGE_EXTS):
            count += 1
    return count


def _resolve_video_dir(base_path, video_id):
    """
    Supports multiple EK frame layouts:
      1) /base_path/P01_04
      2) /base_path/P01/P01_04
      3) /base_path/P01/rgb_frames/P01_04
      4) /base_path/rgb_frames/P01_04
    """
    pid = video_id.split("_")[0]
    candidates = [
        os.path.join(base_path, video_id),
        os.path.join(base_path, pid, video_id),
        os.path.join(base_path, pid, "rgb_frames", video_id),
        os.path.join(base_path, "rgb_frames", video_id),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


class decode_frame_dirs_to_clips(wds.PipelineStage):
    def __init__(
        self,
        annotations,
        video_meta,
        frames_per_clip=16,
        fps=5,
        transform=None,
        anticipation_time_sec=(1.0, 1.0),
        anticipation_point=(1.0, 1.0),
        frame_template="frame_{:010d}.jpg",
        frame_index_offset=1,
    ):
        self.annotations = annotations
        self.video_meta = video_meta
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.transform = transform
        self.anticipation_time = anticipation_time_sec
        self.anticipation_point = anticipation_point
        self.frame_template = frame_template
        self.frame_index_offset = frame_index_offset

    def run(self, src):
        for video_dir in src:
            video_id = os.path.basename(os.path.normpath(video_dir))
            if video_id not in self.annotations:
                logging.info("video_id %s not found in annotations, skipping", video_id)
                continue

            ano = self.annotations[video_id]
            meta = self.video_meta[video_id]

            vfps = float(meta["fps"])
            max_frame_index = int(meta["num_frames"]) - 1

            fpc = self.frames_per_clip
            fstp = max(1, int(round(vfps / self.fps)))
            nframes = int(fpc * fstp)

            start_frames = ano["start_frame"].values
            stop_frames = ano["stop_frame"].values

            for i, (sf, ef) in enumerate(zip(start_frames, stop_frames)):
                labels_verb = int(ano["verb_class"].values[i])
                labels_noun = int(ano["noun_class"].values[i])

                at = _sample_scalar_or_range(self.anticipation_time)
                aframes = int(round(at * vfps))

                ap = _sample_scalar_or_range(self.anticipation_point)
                af = int(sf * ap + (1.0 - ap) * ef - aframes)

                indices = np.arange(af - nframes, af, fstp).astype(np.int64)
                indices[indices < 0] = 0
                indices[indices > max_frame_index] = max_frame_index

                frame_paths = [
                    os.path.join(
                        video_dir,
                        self.frame_template.format(int(idx) + self.frame_index_offset),
                    )
                    for idx in indices
                ]

                try:
                    buffer = np.stack([_read_rgb(p) for p in frame_paths], axis=0)  # [T, H, W, C]
                except Exception as e:
                    logging.info("Encountered exception loading frames for %s: %s", video_id, e)
                    continue

                if self.transform is not None:
                    buffer = self.transform(buffer)

                yield dict(
                    video=buffer,
                    verb=labels_verb,
                    noun=labels_noun,
                    anticipation_time=at,
                )


class ResampledShards(IterableDataset):
    """An iterable dataset yielding a list of input shards (here: frame directories)."""

    def __init__(self, urls, epoch, training):
        super().__init__()
        self.epoch = epoch
        self.training = training
        self.urls = np.array(urls)
        logging.info("Done initializing ResampledShards")

    def __iter__(self):
        if self.training:
            epoch = self.epoch.get_value()
            gen = torch.Generator()
            gen.manual_seed(epoch)
            yield from self.urls[torch.randperm(len(self.urls), generator=gen)]
        else:
            yield from self.urls[torch.arange(len(self.urls))]


def get_video_wds_dataset(
    batch_size,
    input_shards,
    video_decoder,
    training,
    epoch=0,
    world_size=1,
    rank=0,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
):
    assert input_shards is not None
    _, num_shards = get_dataset_size(input_shards)
    logging.info(f"Total number of shards across all data is {num_shards=}")

    epoch = SharedEpoch(epoch=epoch)
    pipeline = [
        ResampledShards(input_shards, epoch=epoch, training=training),
        split_by_node(rank=rank, world_size=world_size),
        wds.split_by_worker,
        video_decoder,
        wds.to_tuple("video", "verb", "noun", "anticipation_time"),
        wds.batched(batch_size, partial=True, collation_fn=torch.utils.data.default_collate),
    ]
    dataset = wds.DataPipeline(*pipeline)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
        worker_init_fn=pl_worker_init_function,
        pin_memory=pin_memory,
    )
    return dataset, DataInfo(dataloader=dataloader, shared_epoch=epoch)


def filter_annotations(
    base_path,
    train_annotations_path,
    val_annotations_path,
    video_info_path=None,
    frame_template="frame_{:010d}.jpg",
    frame_index_offset=1,
    **kwargs,
):
    del frame_template, frame_index_offset  # not needed at annotation-filter time

    if video_info_path is None:
        raise ValueError(
            "For EK frame-folder input, `video_info_path` is required "
            "(e.g. EPIC_100_video_info.csv) so anticipation_time_sec can be converted to frames."
        )

    tdf = pd.read_csv(train_annotations_path)
    vdf = pd.read_csv(val_annotations_path)
    info_df = pd.read_csv(video_info_path)

    if "video_id" not in info_df.columns or "fps" not in info_df.columns:
        raise ValueError(
            f"`video_info_path` must contain columns ['video_id', 'fps']; got {list(info_df.columns)}"
        )

    fps_lookup = {r["video_id"]: float(r["fps"]) for _, r in info_df.iterrows()}
    info_num_frames = {}
    if "num_frames" in info_df.columns:
        info_num_frames = {r["video_id"]: int(r["num_frames"]) for _, r in info_df.iterrows()}

    # 1) remove val actions not present in train
    tactions = set((v, n) for v, n in zip(tdf["verb_class"].values, tdf["noun_class"].values))
    tverbs = set(v for v, _ in tactions)
    tnouns = set(n for _, n in tactions)

    keep_inds = [(v, n) in tactions for v, n in zip(vdf["verb_class"].values, vdf["noun_class"].values)]
    vdf = vdf[keep_inds]

    # 2) remap classes
    verb_classes = {k: i for i, k in enumerate(tverbs)}
    noun_classes = {k: i for i, k in enumerate(tnouns)}
    action_classes = {k: i for i, k in enumerate(tactions)}

    val_verb_classes = set(verb_classes[v] for v in vdf["verb_class"].values)
    val_noun_classes = set(noun_classes[n] for n in vdf["noun_class"].values)
    val_action_classes = set(action_classes[a] for a in zip(vdf["verb_class"].values, vdf["noun_class"].values))

    def build_annotations(df):
        video_dirs, annotations, video_meta = [], {}, {}
        unique_videos = list(dict.fromkeys(df["video_id"].values))

        missing_dirs = 0
        missing_fps = 0
        empty_dirs = 0

        for uv in unique_videos:
            video_dir = _resolve_video_dir(base_path, uv)
            if video_dir is None:
                logging.info("frame dir not found for video_id=%s under base_path=%s", uv, base_path)
                missing_dirs += 1
                continue

            if uv not in fps_lookup:
                logging.info("fps not found in video_info_path for video_id=%s", uv)
                missing_fps += 1
                continue

            num_frames = info_num_frames.get(uv, None)
            if num_frames is None:
                num_frames = _count_image_files(video_dir)

            if num_frames <= 0:
                logging.info("no image frames found in video_dir=%s", video_dir)
                empty_dirs += 1
                continue

            video_dirs.append(video_dir)
            annotations[uv] = df[df["video_id"] == uv].sort_values(by="start_frame")
            video_meta[uv] = {
                "fps": fps_lookup[uv],
                "num_frames": int(num_frames),
            }

        logging.info(
            "EK frame filter: kept=%d missing_dirs=%d missing_fps=%d empty_dirs=%d",
            len(video_dirs),
            missing_dirs,
            missing_fps,
            empty_dirs,
        )
        return video_dirs, annotations, video_meta

    train_annotations = build_annotations(tdf)
    val_annotations = build_annotations(vdf)

    return dict(
        verbs=verb_classes,
        nouns=noun_classes,
        actions=action_classes,
        val_verbs=val_verb_classes,
        val_nouns=val_noun_classes,
        val_actions=val_action_classes,
        train=train_annotations,
        val=val_annotations,
    )


def make_webvid(
    base_path,
    annotations_path,
    batch_size,
    transform,
    frames_per_clip=16,
    fps=5,
    num_workers=8,
    world_size=1,
    rank=0,
    anticipation_time_sec=(1.0, 1.0),
    persistent_workers=True,
    pin_memory=True,
    training=True,
    anticipation_point=(1.0, 1.0),
    frame_template="frame_{:010d}.jpg",
    frame_index_offset=1,
    **kwargs,
):
    del base_path, kwargs  # annotations_path already contains resolved paths + metadata

    paths, annotations, video_meta = annotations_path
    num_clips = sum(len(a) for a in annotations.values())

    if len(paths) == 0 or num_clips == 0:
        raise ValueError(
            f"No usable EK frame-folder clips found. len(paths)={len(paths)}, num_clips={num_clips}."
        )

    if len(paths) < world_size:
        raise ValueError(
            f"Usable video dirs ({len(paths)}) < world_size ({world_size}). "
            "Please reduce GPU/process count while debugging."
        )

    video_decoder = decode_frame_dirs_to_clips(
        annotations=annotations,
        video_meta=video_meta,
        frames_per_clip=frames_per_clip,
        fps=fps,
        transform=transform,
        anticipation_time_sec=anticipation_time_sec,
        anticipation_point=anticipation_point,
        frame_template=frame_template,
        frame_index_offset=frame_index_offset,
    )

    dataset, datainfo = get_video_wds_dataset(
        batch_size=batch_size,
        input_shards=paths,
        epoch=0,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        video_decoder=video_decoder,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        training=training,
    )

    datainfo.dataloader.num_batches = math.ceil(num_clips / max(1, world_size * batch_size))
    datainfo.dataloader.num_samples = num_clips
    return dataset, datainfo.dataloader, datainfo
