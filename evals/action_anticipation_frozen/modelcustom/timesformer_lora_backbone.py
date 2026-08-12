"""TimeSformer backbone adapter with optional LoRA merge.

This module follows the API contract expected by
evals.action_anticipation_frozen.models.init_module.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

LOGGER = logging.getLogger(__name__)


class TimeSformerBackbone(torch.nn.Module):
	"""Frozen TimeSformer wrapper returning patch-token features."""

	def __init__(
		self,
		model: torch.nn.Module,
		*,
		drop_cls_token: bool = True,
		temporal_pool: str = "none",
	) -> None:
		super().__init__()
		self.model = model
		self.drop_cls_token = drop_cls_token
		self.temporal_pool = temporal_pool
		self.embed_dim = int(model.config.hidden_size)

	def forward(self, x: torch.Tensor, anticipation_times: Optional[torch.Tensor] = None) -> torch.Tensor:
		# Input in this repo is [B, C, T, H, W]. HF TimeSformer expects [B, T, C, H, W].
		pixel_values = x.permute(0, 2, 1, 3, 4).contiguous()
		outputs = self.model(pixel_values=pixel_values)
		tokens = outputs.last_hidden_state

		if self.drop_cls_token and tokens.size(1) > 0:
			tokens = tokens[:, 1:, :]

		if self.temporal_pool == "mean":
			tokens = tokens.mean(dim=1, keepdim=True)

		return tokens


def _build_model(pretrain_kwargs: Dict[str, Any]) -> torch.nn.Module:
	from transformers import TimesformerConfig, TimesformerModel

	if pretrain_kwargs.get("from_config", False):
		config_kwargs = pretrain_kwargs.get("timesformer_config", {})
		LOGGER.info("Instantiating TimeSformer from config only")
		return TimesformerModel(TimesformerConfig(**config_kwargs))

	base_model_name = pretrain_kwargs.get("base_model_name")
	base_model_path = pretrain_kwargs.get("base_model_path")
	if not base_model_name and not base_model_path:
		raise ValueError("Either pretrain_kwargs.base_model_name or pretrain_kwargs.base_model_path must be set")

	if base_model_path:
		LOGGER.info("Loading TimeSformer from local path: %s", base_model_path)
		model = TimesformerModel.from_pretrained(base_model_path)
	else:
		LOGGER.info("Loading TimeSformer from model hub id: %s", base_model_name)
		model = TimesformerModel.from_pretrained(base_model_name)

	lora_path = pretrain_kwargs.get("lora_path")
	if lora_path:
		LOGGER.info("Loading LoRA adapter from: %s", lora_path)
		from peft import PeftModel

		model = PeftModel.from_pretrained(model, lora_path)
		if pretrain_kwargs.get("merge_lora", True):
			LOGGER.info("Merging LoRA adapter into TimeSformer backbone")
			model = model.merge_and_unload()

	return model


def init_module(
	frames_per_clip: int,
	frames_per_second: int,
	resolution: int,
	checkpoint: Optional[str],
	model_kwargs: dict,
	wrapper_kwargs: dict,
	**kwargs,
):
	del frames_per_clip, frames_per_second, resolution, checkpoint, kwargs

	pretrain_kwargs = model_kwargs
	backbone = _build_model(pretrain_kwargs)

	module = TimeSformerBackbone(
		model=backbone,
		drop_cls_token=bool(pretrain_kwargs.get("drop_cls_token", True)),
		temporal_pool=str(pretrain_kwargs.get("temporal_pool", "none")),
	)
	return module
