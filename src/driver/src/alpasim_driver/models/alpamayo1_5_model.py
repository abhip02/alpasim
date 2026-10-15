# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Alpamayo 1.5 wrapper implementing the common interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

from ..schema import ModelConfig
from .alpamayo_base import (
    CAMERA_NAME_TO_INDEX,
    AlpamayoBaseModel,
    configure_deterministic_runtime,
)
from .trajectory_selection import TrajectorySelectionStrategy

logger = logging.getLogger(__name__)
_ATTN_IMPLEMENTATION = "sdpa"


class Alpamayo15Model(AlpamayoBaseModel):
    """Alpamayo 1.5 wrapper implementing the common interface.

    Compared to Alpamayo 1, Alpamayo 1.5 adds:
    - Camera index awareness via ``helper.create_message(frames, camera_indices)``
    - Optional classifier-free guidance navigation
      (``sample_trajectories_from_data_with_vlm_rollout_cfg_nav``)
    """

    @classmethod
    def _variant_init_kwargs(cls, model_cfg: ModelConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"cfg_guidance_weight": model_cfg.cfg_guidance_weight}
        if model_cfg.sharding:
            if model_cfg.model_type != "alpamayo1_5":
                raise ValueError(
                    "Sharding is supported only for the Alpamayo 1.5 adapter"
                )
            kwargs["sharding"] = True
        return kwargs

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        camera_ids: list[str],
        context_length: int = AlpamayoBaseModel.DEFAULT_CONTEXT_LENGTH,
        num_traj_samples: int = 1,
        top_p: float = 0.98,
        temperature: float = 0.6,
        cfg_guidance_weight: float | None = None,
        force_determinism: bool = False,
        selection_strategy: TrajectorySelectionStrategy = (
            TrajectorySelectionStrategy.ALWAYS_FIRST
        ),
        max_num_distance_points: int = 64,
        skip_first_n_distance_points: int = 0,
        sharding: bool = False,
    ):
        """Initialize Alpamayo 1.5 model.

        Args:
            checkpoint_path: Path or HuggingFace model ID for Alpamayo 1.5 checkpoint.
            device: Torch device for inference; local cuda:0 when sharding.
            camera_ids: List of camera IDs (supports multiple cameras).
            context_length: Number of temporal frames per camera (default 4).
            num_traj_samples: Number of trajectory samples to generate.
            top_p: Top-p sampling parameter for VLM generation.
            temperature: Temperature for VLM sampling.
            cfg_guidance_weight: Weight for classifier-free guidance on the
                navigation instruction.  Unset leaves guidance to the
                checkpoint's diffusion config; a weight forces it on and blends
                the conditioned and unconditioned velocity fields, which costs a
                second forward pass and roughly 60 GB VRAM (vs ~40 GB).  A weight
                of 1.0 reproduces the conditioned field and buys nothing.
            force_determinism: Whether to make stochastic inference repeatable from
                each prediction's inference seed.
            selection_strategy: How to pick one of the sampled trajectories.
            max_num_distance_points: Waypoints entering the selection distance
                average.
            skip_first_n_distance_points: Leading waypoints excluded from the
                selection distance average.
            sharding: Use the bundled coordinated map across two visible GPUs.
                Each GPU has a 12 GiB resident weight-and-buffer budget;
                activations and caches require additional memory.
        """
        if sharding:
            if device.type != "cuda" or device.index not in (None, 0):
                raise ValueError("Sharding requires input device cuda:0")
            if torch.cuda.device_count() != 2:
                raise ValueError("Sharding requires exactly two visible CUDA GPUs")
            if device.index is None and torch.cuda.current_device() != 0:
                raise ValueError("Sharding requires input device cuda:0")
        if force_determinism:
            configure_deterministic_runtime()
        logger.info("Loading Alpamayo 1.5 checkpoint from %s", checkpoint_path)
        logger.info("Using Alpamayo 1.5 attn_implementation=%s", _ATTN_IMPLEMENTATION)

        placement = {}
        if sharding:
            placement = {
                "device_map": json.loads(
                    Path(__file__).with_name("alpamayo1_5_device_map.json").read_text()
                ),
                "max_memory": {0: "12GiB", 1: "12GiB"},
            }
        model = Alpamayo1_5.from_pretrained(
            checkpoint_path,
            dtype=self.DTYPE,
            attn_implementation=_ATTN_IMPLEMENTATION,
            **placement,
        )
        if sharding:
            for gpu in (0, 1):
                weight_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in list(model.parameters()) + list(model.buffers())
                    if tensor.device == torch.device(f"cuda:{gpu}")
                )
                logger.info(
                    "Shard %s resident weights and buffers: %.3f GiB",
                    gpu,
                    weight_bytes / 2**30,
                )
                if weight_bytes > 12 * 2**30:
                    raise RuntimeError(f"Shard {gpu} exceeds the 12 GiB weight budget")
        else:
            model = model.to(device)
        processor = helper.get_processor(model.tokenizer)

        self._cfg_guidance_weight = cfg_guidance_weight

        self._init_common(
            model=model,
            processor=processor,
            helper_module=helper,
            device=device,
            camera_ids=camera_ids,
            context_length=context_length,
            num_traj_samples=num_traj_samples,
            top_p=top_p,
            temperature=temperature,
            force_determinism=force_determinism,
            selection_strategy=selection_strategy,
            max_num_distance_points=max_num_distance_points,
            skip_first_n_distance_points=skip_first_n_distance_points,
        )

        if cfg_guidance_weight is not None:
            logger.info(
                "Navigation guidance enabled with weight %s (requires ~60 GB VRAM)",
                cfg_guidance_weight,
            )

    def _create_chat_message(
        self, image_frames: torch.Tensor, nav_text: str | None
    ) -> list:
        """Create chat message with camera indices and navigation text."""
        return self._helper.create_message(
            image_frames.flatten(0, 1),
            self._camera_indices(),
            num_frames_per_camera=self._context_length,
            nav_text=nav_text,
        )

    def _camera_indices(self) -> torch.Tensor:
        """Return camera indices in the order used by image preprocessing."""
        sorted_camera_ids = sorted(
            self._camera_ids, key=lambda cam_id: CAMERA_NAME_TO_INDEX[cam_id]
        )
        return torch.tensor(
            [CAMERA_NAME_TO_INDEX[cam_id] for cam_id in sorted_camera_ids]
        )

    def _run_inference(
        self, model_inputs: dict[str, Any], nav_text: str | None
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Run inference, guiding towards the navigation instruction if asked.

        Guidance blends a pass conditioned on the instruction with one that is
        not, so it needs an instruction to guide towards.
        """
        if self._cfg_guidance_weight is None or nav_text is None:
            return super()._run_inference(model_inputs, nav_text)

        return self._model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav(
            data=model_inputs,
            top_p=self._top_p,
            temperature=self._temperature,
            num_traj_samples=self._num_traj_samples,
            max_generation_length=self.MAX_GENERATION_LENGTH,
            diffusion_kwargs={
                "use_classifier_free_guidance": True,
                "inference_guidance_weight": self._cfg_guidance_weight,
            },
            return_extra=True,
        )
