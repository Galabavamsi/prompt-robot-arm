from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "sim" / "sorting_scene.xml"

# These names are the tiny "symbolic world" for the robot.
# The prompt talks about colors; the simulator talks about body names.
# These dictionaries bridge human language to MuJoCo objects.
COLORS = ("red", "blue", "green", "yellow")
BOX_BY_COLOR = {color: f"{color}_box" for color in COLORS}
BIN_BY_COLOR = {color: f"{color}_bin" for color in COLORS}


@dataclass(frozen=True)
class Step:
    """One structured robot task step.

    This is the first important robotics idea in the project:
    do not let the robot execute raw language directly. Convert language
    into a small, explicit action record that the rest of the system can trust.
    """

    action: str
    color: str
    source: str
    target: str


def parse_instruction(instruction: str) -> list[str]:
    """Extract color order from a simple natural-language sorting prompt.

    This is intentionally simple. Later, this function can be replaced by
    an SLM/LLM/VLA planner, but the rest of the robotics pipeline can stay
    almost the same because it consumes a structured plan.
    """
    lowered = instruction.lower()
    order = []
    for token in re.split(r"[^a-z]+", lowered):
        if token in COLORS and token not in order:
            order.append(token)
    return order or ["red", "blue", "green"]


def make_plan(instruction: str) -> list[Step]:
    """Turn a human instruction into robot-readable pick-place steps."""
    color_order = parse_instruction(instruction)
    plan = []
    for color in color_order:
        plan.append(
            Step(
                action="pick_place",
                color=color,
                source=BOX_BY_COLOR[color],
                target=BIN_BY_COLOR[color],
            )
        )
    return plan


def body_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    """Read a body position from MuJoCo's current simulated world state."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def body_quat(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    """Read a body orientation quaternion so we can preserve object rotation."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xquat[body_id].copy()


def set_free_body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    pos: np.ndarray,
    quat: np.ndarray | None = None,
) -> None:
    """Teleport a free body to a new pose.

    In Version 1 this is our simplification: when the fake gripper "holds"
    a box, we directly set the box pose. A real gripper would use contacts,
    force, friction, and a grasp controller instead.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_id = model.body_jntadr[body_id]
    qpos_address = model.jnt_qposadr[joint_id]
    data.qpos[qpos_address : qpos_address + 3] = pos
    data.qvel[model.jnt_dofadr[joint_id] : model.jnt_dofadr[joint_id] + 6] = 0
    if quat is not None:
        data.qpos[qpos_address + 3 : qpos_address + 7] = quat
    mujoco.mj_forward(model, data)


def set_gripper_pose(data: mujoco.MjData, pos: np.ndarray) -> None:
    """Move the mocap gripper marker.

    A MuJoCo mocap body is a kinematic target: the script can place it
    directly in space. This is useful for learning pick-place sequencing
    before introducing robot joints and inverse kinematics.
    """
    data.mocap_pos[0] = pos
    data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])


def interpolate(start: np.ndarray, end: np.ndarray, steps: int):
    """Generate smooth waypoints between two 3D positions."""
    for alpha in np.linspace(0.0, 1.0, steps):
        yield (1.0 - alpha) * start + alpha * end


def step_viewer(model, data, viewer, seconds: float = 0.01) -> None:
    """Advance physics one step and refresh the viewer."""
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(seconds)


def move_gripper(model, data, viewer, end: np.ndarray, frames: int = 70) -> None:
    """Move the fake gripper to a target point over several frames."""
    start = data.mocap_pos[0].copy()
    for pos in interpolate(start, end, frames):
        set_gripper_pose(data, pos)
        step_viewer(model, data, viewer)


def carry_box(model, data, viewer, box_name: str, end: np.ndarray, frames: int = 90) -> None:
    """Move the fake gripper while keeping a box attached underneath it."""
    start = data.mocap_pos[0].copy()
    quat = body_quat(model, data, box_name)
    for gripper_pos in interpolate(start, end, frames):
        set_gripper_pose(data, gripper_pos)
        box_pos = gripper_pos + np.array([0.0, 0.0, -0.075])
        set_free_body_pose(model, data, box_name, box_pos, quat)
        step_viewer(model, data, viewer)


def execute_step(model, data, viewer, step: Step) -> None:
    """Execute one pick-place step using simple task-space waypoints.

    The sequence below is the classic pick-place skill shape:
    approach -> descend -> lift -> transfer -> place -> retreat.
    Later, the same waypoints become targets for IK or MoveIt.
    """
    box_pos = body_pos(model, data, step.source)
    bin_pos = body_pos(model, data, step.target)

    # These offsets are in meters. We keep the gripper above objects first
    # to avoid sweeping sideways through boxes or bins.
    approach_box = box_pos + np.array([0.0, 0.0, 0.22])
    grasp_box = box_pos + np.array([0.0, 0.0, 0.12])
    lift_box = box_pos + np.array([0.0, 0.0, 0.34])
    approach_bin = bin_pos + np.array([0.0, 0.0, 0.34])
    place_bin = bin_pos + np.array([0.0, 0.0, 0.14])

    print(f"Executing: pick {step.color} -> place in {step.target}")
    move_gripper(model, data, viewer, approach_box)
    move_gripper(model, data, viewer, grasp_box, frames=35)
    carry_box(model, data, viewer, step.source, lift_box, frames=45)
    carry_box(model, data, viewer, step.source, approach_bin)
    carry_box(model, data, viewer, step.source, place_bin, frames=45)

    final_box_pos = bin_pos + np.array([0.0, 0.0, 0.09])
    set_free_body_pose(model, data, step.source, final_box_pos, body_quat(model, data, step.source))
    move_gripper(model, data, viewer, approach_bin, frames=35)


def world_state(model, data) -> dict[str, dict[str, list[float]]]:
    """Expose simulator truth as a small digital-twin state dictionary."""
    objects = {}
    for color, box_name in BOX_BY_COLOR.items():
        objects[box_name] = {
            "color": color,
            "position": body_pos(model, data, box_name).round(3).tolist(),
        }
    return objects


def main() -> None:
    """Load the scene, build a plan, execute it, and keep the viewer open."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instruction",
        default="sort red, blue, green",
        help="Example: 'sort yellow, red, blue'",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    # mj_forward computes positions such as data.xpos from the initial qpos.
    mujoco.mj_forward(model, data)

    plan = make_plan(args.instruction)
    print("Instruction:", args.instruction)
    print("Initial world state:", world_state(model, data))
    print("Plan:")
    for index, step in enumerate(plan, start=1):
        print(f"  {index}. {step.action}: {step.source} -> {step.target}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")

        for _ in range(80):
            step_viewer(model, data, viewer)

        for step in plan:
            execute_step(model, data, viewer, step)

        print("Final world state:", world_state(model, data))
        print("Done. Close the viewer window to exit.")
        while viewer.is_running():
            step_viewer(model, data, viewer)


if __name__ == "__main__":
    main()
