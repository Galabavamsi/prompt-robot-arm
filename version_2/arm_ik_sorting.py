from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "sim" / "sorting_arm_scene.xml"

# Symbolic world vocabulary. Prompts mention colors; MuJoCo uses body names.
COLORS = ("red", "blue", "green", "yellow")
BOX_BY_COLOR = {color: f"{color}_box" for color in COLORS}
BIN_BY_COLOR = {color: f"{color}_bin" for color in COLORS}

# These numbers must match the SCARA geometry in sorting_arm_scene.xml.
BASE_XY = np.array([-0.48, 0.0])
BASE_Z = 0.30
LINK1 = 0.38
LINK2 = 0.36
LINK_Z_OFFSET = 0.10
GRASP_SITE_Z_OFFSET = -0.30

# The cube geoms in the XML use size="0.04 0.04 0.04", so the cube half-height
# is 4 cm. We target the grasp site just above the cube top face instead of at
# the cube center. This removes the visual "box inside gripper" offset.
BOX_HALF_HEIGHT = 0.04
GRASP_CLEARANCE = 0.012
GRASP_SITE_ABOVE_BOX_CENTER = BOX_HALF_HEIGHT + GRASP_CLEARANCE
BOX_CENTER_FROM_GRASP_SITE = np.array([0.0, 0.0, -GRASP_SITE_ABOVE_BOX_CENTER])

SCARA_JOINTS = ("base_yaw", "elbow_yaw", "gripper_z")


@dataclass(frozen=True)
class Step:
    """One robot-readable pick-place instruction."""

    action: str
    color: str
    source: str
    target: str


def parse_instruction(instruction: str) -> list[str]:
    """Extract the requested color sequence from a simple prompt."""
    lowered = instruction.lower()
    order = []
    for token in re.split(r"[^a-z]+", lowered):
        if token in COLORS and token not in order:
            order.append(token)
    return order or ["red", "blue", "green"]


def make_plan(instruction: str) -> list[Step]:
    """Convert a prompt into structured robot actions."""
    return [
        Step("pick_place", color, BOX_BY_COLOR[color], BIN_BY_COLOR[color])
        for color in parse_instruction(instruction)
    ]


def mj_id(model: mujoco.MjModel, obj_type, name: str) -> int:
    """Small helper for named MuJoCo lookups."""
    return mujoco.mj_name2id(model, obj_type, name)


def joint_qaddr(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_qposadr[joint_id]


def body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[body_id].copy()


def body_quat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xquat[body_id].copy()


def grasp_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    site_id = mj_id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    return data.site_xpos[site_id].copy()


def set_free_body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    pos: np.ndarray,
    quat: np.ndarray | None = None,
) -> None:
    """Place a free body directly in the world.

    This keeps Version 2 focused on arm IK. A future version should replace
    this manual attachment with real gripper contacts and force control.
    """
    joint_id = mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{body_name}_joint")
    qaddr = model.jnt_qposadr[joint_id]
    daddr = model.jnt_dofadr[joint_id]
    data.qpos[qaddr : qaddr + 3] = pos
    data.qvel[daddr : daddr + 6] = 0.0
    if quat is not None:
        data.qpos[qaddr + 3 : qaddr + 7] = quat
    mujoco.mj_forward(model, data)


def set_target_marker(data: mujoco.MjData, pos: np.ndarray) -> None:
    data.mocap_pos[0] = pos
    data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def current_scara_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return np.array([data.qpos[joint_qaddr(model, name)] for name in SCARA_JOINTS])


def choose_closest_solution(
    current: np.ndarray,
    candidates: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Pick the IK solution with the smallest joint motion from current pose."""
    best = candidates[0]
    best_cost = float("inf")
    for candidate in candidates:
        diff = np.array(candidate) - current
        diff[:2] = (diff[:2] + math.pi) % (2.0 * math.pi) - math.pi
        cost = float(np.linalg.norm(diff))
        if cost < best_cost:
            best = candidate
            best_cost = cost
    return best


def solve_scara_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
) -> tuple[float, float, float]:
    """Analytic inverse kinematics for the teaching SCARA arm.

    SCARA IK is intentionally easy to reason about:
    - solve a 2-link planar arm for X/Y
    - solve a vertical slide for Z

    This is a better first arm for tabletop sorting than a general elbow arm.
    """
    dx, dy = target_pos[:2] - BASE_XY
    r2 = dx * dx + dy * dy

    cos_elbow = (r2 - LINK1 * LINK1 - LINK2 * LINK2) / (2.0 * LINK1 * LINK2)
    cos_elbow = clamp(float(cos_elbow), -1.0, 1.0)

    elbow_options = [math.acos(cos_elbow), -math.acos(cos_elbow)]
    candidates = []
    for elbow in elbow_options:
        shoulder = math.atan2(dy, dx) - math.atan2(
            LINK2 * math.sin(elbow),
            LINK1 + LINK2 * math.cos(elbow),
        )
        slide = target_pos[2] - (BASE_Z + LINK_Z_OFFSET + GRASP_SITE_Z_OFFSET)
        slide = clamp(slide, -0.02, 0.34)
        candidates.append((shoulder, elbow, slide))

    return choose_closest_solution(current_scara_qpos(model, data), candidates)


def set_scara_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: tuple[float, float, float],
) -> None:
    """Write SCARA joint values directly, then recompute kinematics."""
    for joint_name, value in zip(SCARA_JOINTS, qpos):
        data.qpos[joint_qaddr(model, joint_name)] = value
    mujoco.mj_forward(model, data)


def render_frame(model, data, viewer, seconds: float = 0.01) -> None:
    """Refresh the viewer without advancing unstable dynamics.

    This is the key stability change. The first arm version used mj_step(),
    which asked MuJoCo to dynamically simulate a robot whose joints we were
    also teleporting. That can create huge accelerations. Here we teach
    kinematics first: set qpos -> mj_forward -> viewer sync.
    """
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(seconds)


def interpolate(start: np.ndarray, end: np.ndarray, frames: int):
    for alpha in np.linspace(0.0, 1.0, frames):
        yield (1.0 - alpha) * start + alpha * end


def move_arm_to(
    model,
    data,
    viewer,
    target_pos: np.ndarray,
    frames: int = 70,
    carried_box: str | None = None,
) -> None:
    """Move the SCARA gripper through straight-line Cartesian waypoints."""
    start = grasp_pos(model, data)
    quat = body_quat(model, data, carried_box) if carried_box else None

    for waypoint in interpolate(start, target_pos, frames):
        set_target_marker(data, waypoint)
        set_scara_pose(model, data, solve_scara_ik(model, data, waypoint))
        if carried_box:
            set_free_body_pose(
                model,
                data,
                carried_box,
                grasp_pos(model, data) + BOX_CENTER_FROM_GRASP_SITE,
                quat,
            )
        render_frame(model, data, viewer)


def execute_step(model, data, viewer, step: Step) -> None:
    """Execute one pick-place step."""
    box_pos = body_pos(model, data, step.source)
    bin_pos = body_pos(model, data, step.target)

    approach_box = box_pos + np.array([0.0, 0.0, 0.24])
    grasp_box = box_pos + np.array([0.0, 0.0, GRASP_SITE_ABOVE_BOX_CENTER])
    lift_box = box_pos + np.array([0.0, 0.0, 0.28])
    approach_bin = bin_pos + np.array([0.0, 0.0, 0.28])
    place_bin = bin_pos + np.array([0.0, 0.0, 0.09 + GRASP_SITE_ABOVE_BOX_CENTER])

    print(f"Executing with SCARA IK: pick {step.color} -> place in {step.target}")
    move_arm_to(model, data, viewer, approach_box)
    move_arm_to(model, data, viewer, grasp_box, frames=35)
    move_arm_to(model, data, viewer, lift_box, frames=45, carried_box=step.source)
    move_arm_to(model, data, viewer, approach_bin, frames=90, carried_box=step.source)
    move_arm_to(model, data, viewer, place_bin, frames=45, carried_box=step.source)

    final_box_pos = bin_pos + np.array([0.0, 0.0, 0.09])
    set_free_body_pose(model, data, step.source, final_box_pos, body_quat(model, data, step.source))
    move_arm_to(model, data, viewer, approach_bin, frames=35)


def world_state(model, data) -> dict[str, dict[str, list[float]]]:
    return {
        box_name: {
            "color": color,
            "position": body_pos(model, data, box_name).round(3).tolist(),
        }
        for color, box_name in BOX_BY_COLOR.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", default="sort red, blue, green")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    home = np.array([-0.16, 0.0, 0.32])
    set_scara_pose(model, data, solve_scara_ik(model, data, home))

    plan = make_plan(args.instruction)
    print("Instruction:", args.instruction)
    print("Initial world state:", world_state(model, data))
    print("Plan:")
    for index, step in enumerate(plan, start=1):
        print(f"  {index}. {step.action}: {step.source} -> {step.target}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mj_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")

        move_arm_to(model, data, viewer, home, frames=20)

        for step in plan:
            execute_step(model, data, viewer, step)

        print("Final world state:", world_state(model, data))
        print("Done. Close the viewer window to exit.")
        while viewer.is_running():
            render_frame(model, data, viewer)


if __name__ == "__main__":
    main()
