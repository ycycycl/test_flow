import torch
import torch.nn as nn
from box import ConfigBox

from flow_drive.utils.dataset import ClusterStatsRetriever
from flow_drive.utils.normalizer import StateNormalizer, ObservationNormalizer
from flow_drive.utils.train_utils import (
    get_diffuser,
    get_encoder,
    get_noise_scheduler,
    load_trained_models,
    load_checkpoint_directly,
)
from flow_drive.utils.infer_utils import (
    sample_action,
    sample_action_with_speed_and_lateral_offsets,
)
from flow_drive.utils.post_processing import (
    smooth_trajectories_preset,
    bound_speed_and_acceleration,
)


class FlowDrivePlanner(nn.Module):
    def __init__(
        self,
        params: ConfigBox,
        device: str = "cpu",
        ckpt_path: str = "None",
        mlflow_exp_name: str = None,
        load_run_name: str = None,
        load_epoch: int = 0,
    ):
        super().__init__()
        if ckpt_path != "None":
            self.encoder, self.decoder = load_checkpoint_directly(
                params, ckpt_path, device=device
            )
        elif (
            load_run_name is not None and load_epoch > 0 and mlflow_exp_name is not None
        ):
            params, self.encoder, self.decoder = load_trained_models(
                params, mlflow_exp_name, load_run_name, load_epoch, device=device
            )
        else:
            self.encoder = get_encoder(params)
            self.decoder = get_diffuser(params)
        self.encoder = self.encoder.to(device).eval()
        self.decoder = self.decoder.to(device).eval()
        self.params = params
        self.device = device
        self.noise_scheduler = get_noise_scheduler(params)
        self.action_normalizer = StateNormalizer.from_json(params.data_processing)
        self.observation_normalizer = ObservationNormalizer.from_json(
            params.data_processing
        )
        self.cluster_retriever = ClusterStatsRetriever(
            params.data_processing.ego_future_clusters_path
        )

    def _risk_attn_cfg(self):
        return self.params.get("risk_attn", {})

    def _build_risk_attn_neighbor_bias(self, inputs: dict):
        cfg = self._risk_attn_cfg()
        logit_scale = float(cfg.get("logit_scale", 0.0))
        if not bool(cfg.get("enabled", False)) or logit_scale == 0.0:
            return None
        if "neighbor_agents_past" not in inputs or "ego_current_state" not in inputs:
            return None

        neighbors = inputs["neighbor_agents_past"]
        ego = inputs["ego_current_state"]
        agent_num = min(int(self.params.encoder.agent_num), neighbors.shape[1])
        if agent_num == 0:
            return None

        eps = float(cfg.get("eps", 1e-3))
        current = neighbors[:, :agent_num, -1, :8]
        ego_pos = ego[:, None, :2]
        ego_vel = ego[:, None, 4:6]
        neighbor_pos = current[..., :2]
        neighbor_vel = current[..., 4:6]

        valid = current[..., :8].abs().sum(dim=-1) > 0
        rel_pos = neighbor_pos - ego_pos
        rel_vel = neighbor_vel - ego_vel
        current_distance = torch.linalg.norm(rel_pos, dim=-1).clamp_min(eps)

        closing_speed = -(rel_pos * rel_vel).sum(dim=-1) / current_distance
        ttc = current_distance / closing_speed.clamp_min(eps)

        sigma_d = max(float(cfg.get("distance_sigma", 10.0)), eps)
        sigma_t = max(float(cfg.get("ttc_sigma", 4.0)), eps)
        sigma_p = max(float(cfg.get("predicted_distance_sigma", 5.0)), eps)
        distance_weight = float(cfg.get("distance_weight", 1.0))
        ttc_weight = float(cfg.get("ttc_weight", 1.0))
        predicted_distance_weight = float(cfg.get("predicted_distance_weight", 1.0))

        risk = distance_weight * torch.exp(-current_distance / sigma_d)
        ttc_term = torch.where(
            closing_speed > 0,
            torch.exp(-ttc / sigma_t),
            torch.zeros_like(ttc),
        )
        risk = risk + ttc_weight * ttc_term

        horizon_steps = cfg.get("prediction_horizon_steps", None)
        if horizon_steps is None:
            horizon_steps = int(self.params.diffuser.pred_horizon)
        else:
            horizon_steps = int(horizon_steps)

        if predicted_distance_weight != 0.0 and horizon_steps > 0:
            dt = float(cfg.get("dt", 0.1))
            times = (
                torch.arange(
                    1,
                    horizon_steps + 1,
                    device=neighbors.device,
                    dtype=neighbors.dtype,
                )
                * dt
            )
            future_rel_pos = rel_pos[:, :, None, :] + rel_vel[:, :, None, :] * times[
                None, None, :, None
            ]
            future_distance = torch.linalg.norm(future_rel_pos, dim=-1)

            if bool(cfg.get("use_box_clearance", True)):
                neighbor_width = current[..., 6].clamp_min(0.0)
                neighbor_length = current[..., 7].clamp_min(0.0)
                neighbor_radius = 0.5 * torch.sqrt(
                    neighbor_width.square() + neighbor_length.square()
                )
                ego_width = float(cfg.get("ego_width", 2.0))
                ego_length = float(cfg.get("ego_length", 4.8))
                ego_radius = 0.5 * (ego_width**2 + ego_length**2) ** 0.5
                future_distance = (
                    future_distance - (ego_radius + neighbor_radius)[:, :, None]
                ).clamp_min(0.0)

            predicted_min_distance = future_distance.min(dim=-1).values
            risk = risk + predicted_distance_weight * torch.exp(
                -predicted_min_distance / sigma_p
            )

        risk = risk.masked_fill(~valid, 0.0).clamp_min(0.0)
        risk_clip = cfg.get("risk_clip", None)
        if risk_clip is not None:
            risk = risk.clamp(max=float(risk_clip))

        return risk * logit_scale

    def _attach_risk_attn_bias(self, encoder_outputs: dict, neighbor_bias):
        if neighbor_bias is None:
            return encoder_outputs

        B, token_num, _ = encoder_outputs["encoding"].shape
        full_bias = torch.zeros(
            (B, token_num),
            device=encoder_outputs["encoding"].device,
            dtype=encoder_outputs["encoding"].dtype,
        )
        neighbor_num = min(neighbor_bias.shape[1], token_num)
        full_bias[:, :neighbor_num] = neighbor_bias[:, :neighbor_num].to(
            device=full_bias.device, dtype=full_bias.dtype
        )
        full_bias = full_bias.masked_fill(encoder_outputs["mask"], 0.0)
        encoder_outputs["attn_bias"] = full_bias
        return encoder_outputs

    def forward(self, inputs: dict):
        with torch.inference_mode():
            risk_attn_neighbor_bias = self._build_risk_attn_neighbor_bias(inputs)
            inputs = self.observation_normalizer(inputs)

            encoder_outputs = self.encoder(inputs)
            encoder_outputs = self._attach_risk_attn_bias(
                encoder_outputs, risk_attn_neighbor_bias
            )

            action_norm, _ = sample_action(
                self.params,
                self.decoder,
                encoder_outputs,
                self.noise_scheduler,
                inputs["ego_current_state"][:, :4],  # [B, 4], x, y, cos, sin
            )  # [B, future_len, 3 or 4]

            action = self.action_normalizer.inverse(action_norm)

            # transform cos and sin to heading
            heading = torch.arctan2(action[:, :, 3], action[:, :, 2])
            action = torch.cat(
                [action[:, :, :2], heading.unsqueeze(-1)], dim=-1
            )  # [B, future_len, 3], x, y, yaw
        return action

    def plan_multiple_trajectories_with_moderated_offset(
        self,
        inputs: dict,
        speed_limit,  # [B], normalized speed limit between [0, 1]
        speed_offsets,
        lateral_offsets,
    ) -> torch.Tensor:
        """
        Plan multiple trajectories from the same input and return the one with the maximum reward.
        """
        with torch.inference_mode():
            ego_state_unnormalized = inputs[
                "ego_current_state"
            ].clone()  # [B, 4], x, y, cos, sin
            risk_attn_neighbor_bias = self._build_risk_attn_neighbor_bias(inputs)
            inputs = self.observation_normalizer(inputs)

            encoder_outputs = self.encoder(inputs)
            encoder_outputs = self._attach_risk_attn_bias(
                encoder_outputs, risk_attn_neighbor_bias
            )

            B, N, D = encoder_outputs["encoding"].shape
            T = self.params.diffuser.pred_horizon
            A = 4

            action_norm, _ = sample_action_with_speed_and_lateral_offsets(
                self.params,
                self.decoder,
                encoder_outputs,
                self.noise_scheduler,
                inputs["ego_current_state"][:, :6],  # [B, 4], x, y, cos, sin, vx, vy
                speed_offsets=speed_offsets,
                lateral_offsets=lateral_offsets,
            )  # [S * B, future_len, 3 or 4]

            actions = self.action_normalizer.inverse(action_norm)  # [20 * B, T, A]
            actions = actions.view(-1, B, T, A)

            # transform cos and sin to heading
            heading = torch.arctan2(actions[:, :, :, 3], actions[:, :, :, 2])
            actions = torch.cat([actions[:, :, :, :2], heading.unsqueeze(-1)], dim=-1)

            # Add current ego state before smoothing and then remove it after
            # Convert ego_state_unnormalized to position format [x, y, yaw]
            ego_current_position = torch.stack(
                [
                    ego_state_unnormalized[:, 0],  # x
                    ego_state_unnormalized[:, 1],  # y
                    torch.atan2(
                        ego_state_unnormalized[:, 3], ego_state_unnormalized[:, 2]
                    ),  # yaw
                ],
                dim=-1,
            )  # [B, 3]

            # Expand to all trajectories and add as first timestep
            S = actions.shape[0]  # Number of sampled trajectories
            ego_current_expanded = ego_current_position[None, :, None, :].expand(
                S, -1, 1, -1
            )  # [S, B, 1, 3]
            actions_with_current = torch.cat(
                [ego_current_expanded, actions], dim=2
            )  # [S, B, T+1, 3]

            # Options: "default", "light", "medium", "strong", "adaptive", "high_quality"
            actions = smooth_trajectories_preset(actions_with_current, preset="strong")[
                :, :, 1:, :
            ]  # [S, B, T, 3]

            actions = bound_speed_and_acceleration(
                actions, ego_state_unnormalized, speed_limit
            )
        return actions
