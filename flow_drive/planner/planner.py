import os
import warnings
import torch
import numpy as np
import numpy.typing as npt
from shapely.geometry import Point
from typing import Deque, Dict, List, Type, Optional, Tuple
import uuid
import glob
import json

warnings.filterwarnings("ignore")

from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.observation.observation_type import (
    Observation,
    DetectionsTracks,
)
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    transform_predictions_to_states,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
)

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    normalize_angle,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.route_utils import (
    route_roadblock_correction,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import (
    Dijkstra,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_emergency_brake import (
    PDMEmergencyBrake,
)

from flow_drive.model.flow_drive_planner import FlowDrivePlanner
from flow_drive.data_process.data_processor import DataProcessor
from flow_drive.utils.train_utils import load_params, set_seed
from flow_drive.utils.plot_dataset_scenario import plot_scenario


def outputs_to_trajectory(
    outputs: torch.Tensor,
    ego_state_history: Deque[EgoState],
    future_horizon: float,
    step_interval: float,
) -> List[InterpolatableState]:
    predictions = outputs[0].detach().cpu().numpy().astype(np.float64)  # [T, 3]
    # transform relative poses to absolute poses
    states = transform_predictions_to_states(
        predictions, ego_state_history, future_horizon, step_interval
    )
    return states


class FlowDrivePlannerWrapper(AbstractPlanner):
    def __init__(
        self,
        device: str = "cpu",
        mlflow_exp_name: str = "None",
        ckpt_path: str = "None",
        load_run_name: str = None,
        load_epoch: int = 0,
        post_mode: int = 0,
        render: bool = False,
        video_dir: str = None,
        emergency_brake_enabled: bool = True,
        diagnostic_dir: str = None,
    ):
        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"

        self._master_seed = 520  # master seed for reproducibility
        self._device = device
        self._post_process = post_mode
        self._ckpt_path = ckpt_path
        # print(f"Post-process: {self._post_process}")

        self._lateral_offset = [0, 6.5 / 20, -6.5 / 20, 12.5 / 20, -12.5 / 20]
        self._speed_offsets = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self._planner = None
        self._planner_id = str(uuid.uuid4())[:8]
        self._mlflow_exp_name = mlflow_exp_name
        self._load_run_name = load_run_name
        self._load_epoch = load_epoch
        self._future_horizon = None
        self._step_interval = 0.1
        self._data_processor = None

        self._map_api: Optional[AbstractMap] = None
        self._route_roadblock_ids = None
        self._trajectory_scorer = TrajectoryScorer(
            emergency_brake_enabled=emergency_brake_enabled
        )
        self._last_selector_tiebreak_info = None

        self._params = load_params(
            os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
        )

        self._render = render
        self._video_dir = video_dir
        self._diagnostic_dir = diagnostic_dir
        self._initial_condition_recorded = False
        self._diagnostic_selector_path = None
        self._diagnostic_initial_path = None
        if self._diagnostic_dir:
            os.makedirs(self._diagnostic_dir, exist_ok=True)
            self._diagnostic_selector_path = os.path.join(
                self._diagnostic_dir, f"selector_frames_{self._planner_id}.jsonl"
            )
            self._diagnostic_initial_path = os.path.join(
                self._diagnostic_dir, f"initial_condition_{self._planner_id}.json"
            )

    def name(self) -> str:
        """
        Inherited.
        """
        return "flow_drive"

    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._iteration = 0
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._trajectory_scorer.initialize(self._map_api, self._route_roadblock_ids)

        self._planner = FlowDrivePlanner(
            self._params,
            self._device,
            self._ckpt_path,
            self._mlflow_exp_name,
            self._load_run_name,
            self._load_epoch,
        )
        self._data_processor = DataProcessor(self._planner.params.data_processing)

        self._future_horizon = self._planner.params.data_processing.future_time_horizon
        self._initialization = initialization

    def planner_input_to_model_inputs(
        self, planner_input: PlannerInput
    ) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self._data_processor.observation_adapter(
            history,
            traffic_light_data,
            self._map_api,
            self._route_roadblock_ids,
            self._device,
        )
        return model_inputs

    def _select_post_process_plan_index(self, scores: npt.ArrayLike) -> int:
        scores_array = np.asarray(scores, dtype=np.float64)
        selected_index = int(np.argmax(scores_array))
        all_scores_zero = bool(np.allclose(scores_array, 0.0, atol=1e-12, rtol=0.0))
        info = {
            "selected_index": selected_index,
            "original_argmax_index": selected_index,
            "all_scores_zero": all_scores_zero,
            "allzero_stop_applied": False,
            "selector_action": "argmax",
            "tiebreak_applied": False,
            "safe_candidate_count": None,
        }

        if all_scores_zero:
            selected_index = -1
            info["selected_index"] = selected_index
            info["allzero_stop_applied"] = True
            info["selector_action"] = "kinematic_stop"
            info["tiebreak_applied"] = True

        self._last_selector_tiebreak_info = info
        return selected_index

    def _write_diagnostic_jsonl(self, path: str, record: Dict) -> None:
        if not path:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _record_selector_diagnostic(self, scores: npt.ArrayLike, selected_index: int) -> None:
        if not self._diagnostic_selector_path:
            return
        scores_array = np.asarray(scores, dtype=np.float64)
        record = {
            "iteration": int(self._iteration),
            "selected_index": int(selected_index),
            "scores": scores_array.tolist(),
            "all_scores_zero": bool(np.allclose(scores_array, 0.0, atol=1e-12, rtol=0.0)),
            "selector_action": (
                self._last_selector_tiebreak_info or {}
            ).get("selector_action", "argmax"),
            "allzero_stop_applied": bool(
                (self._last_selector_tiebreak_info or {}).get("allzero_stop_applied", False)
            ),
        }
        self._write_diagnostic_jsonl(self._diagnostic_selector_path, record)

    def _record_initial_condition_diagnostic(self, current_input: PlannerInput) -> None:
        if self._initial_condition_recorded or not self._diagnostic_initial_path:
            return
        self._initial_condition_recorded = True
        ego_state, _ = current_input.history.current_state
        record = {
            "iteration": int(self._iteration),
            "rear_axle": {
                "x": float(ego_state.rear_axle.x),
                "y": float(ego_state.rear_axle.y),
                "heading": float(ego_state.rear_axle.heading),
            },
            "speed_mps": float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x),
            "rear_axle_in_drivable_area": None,
            "ego_footprint_intersects_drivable_area": None,
            "drivable_area_tokens": [],
            "error": None,
        }
        try:
            drivable_area_map = get_drivable_area_map(
                self._map_api, ego_state, self._trajectory_scorer._map_radius
            )
            rear_axle_tokens = list(drivable_area_map.intersects(Point(*ego_state.rear_axle.array)))
            footprint_tokens = list(drivable_area_map.intersects(ego_state.car_footprint.geometry))
            record["rear_axle_in_drivable_area"] = bool(rear_axle_tokens)
            record["ego_footprint_intersects_drivable_area"] = bool(footprint_tokens)
            record["drivable_area_tokens"] = sorted(set(map(str, rear_axle_tokens + footprint_tokens)))
        except Exception as exc:
            record["error"] = repr(exc)
        with open(self._diagnostic_initial_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    def _build_stop_ego_states(self, ego_state: EgoState) -> List[EgoState]:
        horizon = float(self._future_horizon or 4.0)
        step_interval = float(self._step_interval)
        num_steps = max(1, int(round(horizon / step_interval)))
        start_time_us = int(ego_state.time_us)
        dt_us = int(round(step_interval * 1e6))
        rear_axle = ego_state.rear_axle
        vehicle_parameters = ego_state.car_footprint.vehicle_parameters
        is_in_auto_mode = bool(getattr(ego_state, "is_in_auto_mode", True))
        wheel_base = float(getattr(vehicle_parameters, "wheel_base", 3.089))
        max_brake_deceleration = 3.8  # [m/s^2], matches the local post-process bound.
        stop_speed_threshold = 0.05

        x = float(rear_axle.x)
        y = float(rear_axle.y)
        heading = float(rear_axle.heading)
        steering_angle = float(ego_state.tire_steering_angle)

        velocity_2d = ego_state.dynamic_car_state.rear_axle_velocity_2d
        signed_speed = float(velocity_2d.x)
        abs_speed = abs(signed_speed)
        direction = 1.0 if signed_speed >= 0.0 else -1.0
        deceleration = (
            max_brake_deceleration if abs_speed > stop_speed_threshold else 0.0
        )
        signed_acceleration = -direction * deceleration if deceleration > 0.0 else 0.0

        states: List[EgoState] = [ego_state]
        previous_angular_velocity = float(
            getattr(ego_state.dynamic_car_state, "angular_velocity", 0.0)
        )
        speed = abs_speed

        for step_idx in range(1, num_steps + 1):
            if speed <= stop_speed_threshold or deceleration <= 0.0:
                next_speed = 0.0
                average_signed_speed = 0.0
                angular_velocity = 0.0
                angular_acceleration = -previous_angular_velocity / step_interval
                acceleration_2d = StateVector2D(0.0, 0.0)
            else:
                next_speed = max(0.0, speed - deceleration * step_interval)
                average_signed_speed = direction * 0.5 * (speed + next_speed)
                angular_velocity = (
                    direction * next_speed * np.tan(steering_angle) / wheel_base
                    if wheel_base > 0.0
                    else 0.0
                )
                angular_acceleration = (
                    angular_velocity - previous_angular_velocity
                ) / step_interval
                acceleration_2d = StateVector2D(signed_acceleration, 0.0)

            average_angular_velocity = 0.5 * (
                previous_angular_velocity + angular_velocity
            )
            heading_midpoint = heading + 0.5 * average_angular_velocity * step_interval
            x += average_signed_speed * np.cos(heading_midpoint) * step_interval
            y += average_signed_speed * np.sin(heading_midpoint) * step_interval
            heading = normalize_angle(heading + average_angular_velocity * step_interval)

            states.append(
                EgoState.build_from_rear_axle(
                    rear_axle_pose=StateSE2(x, y, heading),
                    rear_axle_velocity_2d=StateVector2D(direction * next_speed, 0.0),
                    rear_axle_acceleration_2d=acceleration_2d,
                    tire_steering_angle=steering_angle,
                    time_point=TimePoint(start_time_us + step_idx * dt_us),
                    vehicle_parameters=vehicle_parameters,
                    is_in_auto_mode=is_in_auto_mode,
                    angular_vel=angular_velocity,
                    angular_accel=angular_acceleration,
                    tire_steering_rate=0.0,
                )
            )
            previous_angular_velocity = angular_velocity
            speed = next_speed

        return states

    def _ego_states_to_relative_plan(
        self,
        ego_states: List[EgoState],
        anchor_ego_state: EgoState,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        relative_plan = np.zeros(shape, dtype=np.float64)
        anchor = anchor_ego_state.rear_axle
        anchor_x = float(anchor.x)
        anchor_y = float(anchor.y)
        anchor_heading = float(anchor.heading)
        cos_heading = np.cos(anchor_heading)
        sin_heading = np.sin(anchor_heading)
        future_states = (
            ego_states[1:]
            if ego_states and ego_states[0] is anchor_ego_state
            else ego_states
        )

        for idx, state in enumerate(future_states[: shape[0]]):
            rear = state.rear_axle
            dx = float(rear.x) - anchor_x
            dy = float(rear.y) - anchor_y
            relative_plan[idx, 0] = cos_heading * dx + sin_heading * dy
            relative_plan[idx, 1] = -sin_heading * dx + cos_heading * dy
            relative_plan[idx, 2] = normalize_angle(float(rear.heading) - anchor_heading)

        return relative_plan

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """
        Inherited.
        """
        set_seed(self._master_seed + self._iteration)  # Set seed for reproducibility
        self._record_initial_condition_diagnostic(current_input)

        inputs = self.planner_input_to_model_inputs(current_input)

        if self._post_process == 1:
            current_lane = self._trajectory_scorer.prepare_scoring(current_input)
            speed_limit = current_lane.speed_limit_mps
            if speed_limit is None:
                speed_limit = 15.0
            speed_limit = torch.tensor(
                [speed_limit], device=self._device
            )  # [B] normalized speed limit
            outputs = self._planner.plan_multiple_trajectories_with_moderated_offset(
                inputs, speed_limit, self._speed_offsets, self._lateral_offset
            )  # (S, B, T, [x, y, heading]), B = 1
            scores, ego_states_list = self._trajectory_scorer.score_plans(outputs)
            index = self._select_post_process_plan_index(scores)
            self._record_selector_diagnostic(scores, index)
            if index < 0:
                ego_state, _ = current_input.history.current_state
                ego_states = self._build_stop_ego_states(ego_state)
            else:
                ego_states = ego_states_list[index]
            trajectory = InterpolatedTrajectory(trajectory=ego_states)
        else:
            outputs = self._planner(inputs)
            trajectory = InterpolatedTrajectory(
                trajectory=outputs_to_trajectory(
                    outputs,
                    current_input.history.ego_states,
                    self._future_horizon,
                    self._step_interval,
                )
            )

        if self._render:
            inputs_for_plotting = {
                k: v[0].cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            if self._post_process > 0:
                if index < 0:
                    ego_state, _ = current_input.history.current_state
                    plan_shape = outputs[0, 0].cpu().numpy().shape
                    inputs_for_plotting["ego_plan"] = self._ego_states_to_relative_plan(
                        ego_states, ego_state, plan_shape
                    )
                else:
                    inputs_for_plotting["ego_plan"] = outputs[index, 0].cpu().numpy()
            else:
                inputs_for_plotting["ego_plan"] = outputs[0].cpu().numpy()
            fig_dir = os.path.join(self._video_dir, self._planner_id)
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = os.path.join(fig_dir, f"{self._iteration}.png")
            plot_scenario(inputs_for_plotting, fig_path)

        self._iteration += 1
        return trajectory

    # make video when deleting the planner
    def __del__(self):
        if self._render:
            img_paths = sorted(
                glob.glob(os.path.join(self._video_dir, self._planner_id, "*.png")),
                key=lambda x: int(os.path.basename(x).split(".")[0]),
            )
            if len(img_paths) > 0:
                import imageio

                video_path = os.path.join(self._video_dir, f"{self._planner_id}.mp4")
                with imageio.get_writer(video_path, fps=10) as video_writer:
                    for img_path in img_paths:
                        image = imageio.v2.imread(img_path)
                        video_writer.append_data(image)
                print(f"Saved video to {video_path}")


class TrajectoryScorer:
    def __init__(self, emergency_brake_enabled: bool = True):
        self._emergency_brake_enabled = emergency_brake_enabled
        self._iteration: int = 0
        self._map_radius: int = 50  # [m]
        self._map_api: Optional[AbstractMap] = None
        self._route_roadblock_ids = None
        self._route_roadblock_dict: Optional[Dict[str, RoadBlockGraphEdgeMapObject]] = (
            None
        )
        self._route_lane_dict: Optional[Dict[str, LaneGraphEdgeMapObject]] = None
        self._centerline: Optional[PDMPath] = None
        self._drivable_area_map: Optional[PDMOccupancyMap] = None
        trajectory_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self._observation = PDMObservation(
            trajectory_sampling, proposal_sampling, self._map_radius
        )
        self._simulator = PDMSimulator(proposal_sampling)
        self._scorer = PDMScorer(proposal_sampling)
        self._emergency_brake = PDMEmergencyBrake(trajectory_sampling)
        self._current_input: Optional[PlannerInput] = None

    def _load_route_dicts(self, route_roadblock_ids: List[str]) -> None:
        """
        Loads roadblock and lane dictionaries of the target route from the map-api.
        :param route_roadblock_ids: ID's of on-route roadblocks
        """
        # remove repeated ids while remaining order in list
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))
        self._route_roadblock_dict = {}
        self._route_lane_dict = {}
        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )
            self._route_roadblock_dict[block.id] = block
            for lane in block.interior_edges:
                self._route_lane_dict[lane.id] = lane

    def _route_roadblock_correction(self, ego_state: EgoState) -> None:
        """
        Corrects the roadblock route and reloads lane-graph dictionaries.
        :param ego_state: state of the ego vehicle.
        """
        route_roadblock_ids = route_roadblock_correction(
            ego_state, self._map_api, self._route_roadblock_dict
        )
        self._load_route_dicts(route_roadblock_ids)

    def _get_discrete_centerline(
        self, current_lane: LaneGraphEdgeMapObject, search_depth: int = 30
    ) -> List[StateSE2]:
        """
        Applies a Dijkstra search on the lane-graph to retrieve discrete centerline.
        :param current_lane: lane object of starting lane.
        :param search_depth: depth of search (for runtime), defaults to 30
        :return: list of discrete states on centerline (x,y,θ)
        """

        roadblocks = list(self._route_roadblock_dict.values())
        roadblock_ids = list(self._route_roadblock_dict.keys())

        # find current roadblock index
        start_idx = np.argmax(
            np.array(roadblock_ids) == current_lane.get_roadblock_id()
        )
        roadblock_window = roadblocks[start_idx : start_idx + search_depth]

        graph_search = Dijkstra(current_lane, list(self._route_lane_dict.keys()))
        route_plan, path_found = graph_search.search(roadblock_window[-1])

        centerline_discrete_path: List[StateSE2] = []
        for lane in route_plan:
            centerline_discrete_path.extend(lane.baseline_path.discrete_path)

        return centerline_discrete_path

    def _get_starting_lane(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        """
        Returns the most suitable starting lane, in ego's vicinity.
        :param ego_state: state of ego-vehicle
        :return: lane object (on-route)
        """
        starting_lane: LaneGraphEdgeMapObject = None
        on_route_lanes, heading_error = self._get_intersecting_lanes(ego_state)

        if on_route_lanes:
            # 1. Option: find lanes from lane occupancy-map
            # select lane with lowest heading error
            starting_lane = on_route_lanes[np.argmin(np.abs(heading_error))]
            return starting_lane

        else:
            # 2. Option: find any intersecting or close lane on-route
            closest_distance = np.inf
            for edge in self._route_lane_dict.values():
                if edge.contains_point(ego_state.center):
                    starting_lane = edge
                    break

                distance = edge.polygon.distance(ego_state.car_footprint.geometry)
                if distance < closest_distance:
                    starting_lane = edge
                    closest_distance = distance

        return starting_lane

    def _get_intersecting_lanes(
        self, ego_state: EgoState
    ) -> Tuple[List[LaneGraphEdgeMapObject], List[float]]:
        """
        Returns on-route lanes and heading errors where ego-vehicle intersects.
        :param ego_state: state of ego-vehicle
        :return: tuple of lists with lane objects and heading errors [rad].
        """
        assert (
            self._drivable_area_map
        ), "AbstractPDMPlanner: Drivable area map must be initialized first!"

        ego_position_array: npt.NDArray[np.float64] = ego_state.rear_axle.array
        ego_rear_axle_point: Point = Point(*ego_position_array)
        ego_heading: float = ego_state.rear_axle.heading

        intersecting_lanes = self._drivable_area_map.intersects(ego_rear_axle_point)

        on_route_lanes, on_route_heading_errors = [], []
        for lane_id in intersecting_lanes:
            if lane_id in self._route_lane_dict.keys():
                # collect baseline path as array
                lane_object = self._route_lane_dict[lane_id]
                lane_discrete_path: List[StateSE2] = (
                    lane_object.baseline_path.discrete_path
                )
                lane_state_se2_array = np.array(
                    [state.array for state in lane_discrete_path], dtype=np.float64
                )
                # calculate nearest state on baseline
                lane_distances = (
                    ego_position_array[None, ...] - lane_state_se2_array
                ) ** 2
                lane_distances = lane_distances.sum(axis=-1) ** 0.5

                # calculate heading error
                heading_error = (
                    lane_discrete_path[np.argmin(lane_distances)].heading - ego_heading
                )
                heading_error = np.abs(normalize_angle(heading_error))

                # add lane to candidates
                on_route_lanes.append(lane_object)
                on_route_heading_errors.append(heading_error)

        return on_route_lanes, on_route_heading_errors

    def initialize(self, map_api: AbstractMap, route_roadblock_ids: List[str]) -> None:
        """
        Initializes the trajectory scorer with map API and route roadblock IDs.
        :param map_api: map API to access map objects
        :param route_roadblock_ids: list of roadblock IDs on the route
        """
        self._map_api = map_api
        self._route_roadblock_ids = route_roadblock_ids
        self._load_route_dicts(route_roadblock_ids)

    def prepare_scoring(self, current_input: PlannerInput) -> LaneGraphEdgeMapObject:
        ego_state, observation = current_input.history.current_state

        if self._iteration == 0:
            self._route_roadblock_correction(ego_state)

        # Update/Create drivable area polygon map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )

        current_lane = self._get_starting_lane(ego_state)
        self._centerline = PDMPath(self._get_discrete_centerline(current_lane))

        self._current_input = current_input

        return current_lane

    def score_plans(
        self, planner_outputs: torch.Tensor
    ) -> Tuple[np.ndarray, List[List[InterpolatableState]]]:
        """
        planner_outputs: (S, B, T, A) where S = number of plans, B = batch size (1)
        """
        plans_list = []
        ego_states_list = []
        for i in range(planner_outputs.shape[0]):
            ego_states = outputs_to_trajectory(
                planner_outputs[i], self._current_input.history.ego_states, 4.0, 0.1
            )
            plan_state_array = ego_states_to_state_array(ego_states)  # (40, 11)
            plans_list.append(plan_state_array)
            ego_states_list.append(ego_states)
        plans_array = np.array(plans_list)

        ego_state, _ = self._current_input.history.current_state

        # Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(
            plans_array, ego_state
        )

        # Score proposals
        proposal_scores = self._scorer.score_proposals(
            simulated_proposals_array,
            ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )  # [S * B] where S = number of plans, B = batch size (1)
        # Apply brake if emergency is expected
        if self._emergency_brake_enabled:
            trajectory = self._emergency_brake.brake_if_emergency(
                ego_state, proposal_scores, self._scorer
            )
            if trajectory is not None:
                ego_states_list = [trajectory.get_sampled_trajectory()] * len(
                    ego_states_list
                )

        self._iteration += 1
        return proposal_scores, ego_states_list
