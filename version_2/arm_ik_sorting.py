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
SCENE_PATH = ROOT / "sim" / "sorting_arm_scene.xml"

# Same symbolic layer as Version 1: prompts mention colors, but MuJoCo
# needs exact body names. The robot code should work with structured names.
COLORS = ("red", "blue", "green", "yellow")
BOX_BY_COLOR = {color: f"{color}_box" for color in COLORS}
BIN_BY_COLOR = {color: f"{color}_bin" for color in COLORS}

# These are the joints our simple IK solver is allowed to move.
# Real arms usually have 6 or 7 joints; this small arm is for learning.
ARM_JOINTS = ("base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch")


@dataclass(frozen=True)
class Step:
    """One robot-readable task step produced from a human instruction."""

    action: str
    color: str
    source: str
    target: str


def parse_instruction(instruction: str) -> list[str]:
    """Extract the requested color order from a basic prompt."""
    lowered = instruction.lower()
    order = []
    for token in re.split(r"[^a-z]+", lowered):
        if token in COLORS and token not in order:
            order.append(token)
    return order or ["red", "blue", "green"]


def make_plan(instruction: str) -> list[Step]:
    """Create pick-place steps from the parsed color order."""
    return [
        Step("pick_place", color, BOX_BY_COLOR[color], BIN_BY_COLOR[color])
        for color in parse_instruction(instruction)
    ]


def body_id(model: mujoco.MjModel, name: str) -> int:
    """Look up a MuJoCo body id by name."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def joint_id(model: mujoco.MjModel, name: str) -> int:
    """Look up a MuJoCo joint id by name."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def site_id(model: mujoco.MjModel, name: str) -> int:
    """Look up a MuJoCo site id by name.

    Sites are marker points attached to bodies. Here, grasp_site is the
    tool-center point we want IK to move to each target.
    """
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """Read the current world-space position of a body."""
    return data.xpos[body_id(model, name)].copy()


def body_quat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """Read the current world-space orientation of a body."""
    return data.xquat[body_id(model, name)].copy()


def controlled_addresses(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    """Find qpos/dof addresses for the joints controlled by IK.

    MuJoCo stores all joint positions and velocities in large arrays.
    A joint name must be translated into array addresses before we can
    read or update that joint.
    """
    qpos_addresses = []
    dof_addresses = []
    for name in ARM_JOINTS:
        jid = joint_id(model, name)
        qpos_addresses.append(model.jnt_qposadr[jid])
        dof_addresses.append(model.jnt_dofadr[jid])
    return qpos_addresses, dof_addresses


def clamp_arm_joints(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Keep IK from pushing joints outside their declared limits."""
    for name in ARM_JOINTS:
        jid = joint_id(model, name)
        qaddr = model.jnt_qposadr[jid]
        lower, upper = model.jnt_range[jid]
        data.qpos[qaddr] = np.clip(data.qpos[qaddr], lower, upper)


def set_free_body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    pos: np.ndarray,
    quat: np.ndarray | None = None,
) -> None:
    """Set a box pose directly while it is being carried.

    This version teaches arm IK, not gripper contact physics. So the box is
    still attached manually while carried. A later version should replace this
    with gripper joints, contact constraints, and force/friction behavior.
    """
    jid = joint_id(model, f"{body_name}_joint")
    qaddr = model.jnt_qposadr[jid]
    daddr = model.jnt_dofadr[jid]
    data.qpos[qaddr : qaddr + 3] = pos
    data.qvel[daddr : daddr + 6] = 0.0
    if quat is not None:
        data.qpos[qaddr + 3 : qaddr + 7] = quat
    mujoco.mj_forward(model, data)


def set_target_marker(data: mujoco.MjData, pos: np.ndarray) -> None:
    """Move the translucent target marker so we can see the IK goal."""
    data.mocap_pos[0] = pos
    data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    max_iters: int = 80,
    tolerance: float = 0.004,
    damping: float = 0.02,
    step_scale: float = 0.55,
) -> float:
    """Position-only damped least-squares IK for the grasp site.

    IK asks: what joint angles put the gripper at target_pos?

    This solver repeats:
      1. measure current gripper position
      2. compute error = target - current
      3. ask MuJoCo for the Jacobian
      4. update joint angles in the direction that reduces error

    It controls position only, not gripper orientation. That keeps the first
    IK lesson small enough to understand.
    """
    grasp_sid = site_id(model, "grasp_site")
    qpos_addresses, dof_addresses = controlled_addresses(model)

    for _ in range(max_iters):
        # Recompute all derived kinematic quantities after any qpos change.
        mujoco.mj_forward(model, data)
        current_pos = data.site_xpos[grasp_sid].copy()
        error = target_pos - current_pos
        error_norm = float(np.linalg.norm(error))
        if error_norm < tolerance:
            return error_norm

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, grasp_sid)

        # jacp maps joint velocity to gripper linear velocity. We keep only
        # the columns for the arm joints, ignoring free-box joints.
        jac = jacp[:, dof_addresses]

        # Damping prevents unstable jumps when the arm is near singular or
        # when the target is hard to reach.
        regularizer = damping * np.eye(3)
        delta = jac.T @ np.linalg.solve(jac @ jac.T + regularizer, error)
        for qaddr, dq in zip(qpos_addresses, delta):
            data.qpos[qaddr] += step_scale * dq

        clamp_arm_joints(model, data)

    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(target_pos - data.site_xpos[grasp_sid]))


def step_viewer(model, data, viewer, seconds: float = 0.01) -> None:
    """Advance the simulator one step and redraw the viewer."""
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(seconds)


def move_arm_to(
    model,
    data,
    viewer,
    target_pos: np.ndarray,
    frames: int = 60,
    carried_box: str | None = None,
) -> None:
    """Move the arm end effector through a straight-line target path.

    This is still a simplification. A production system would plan a
    collision-aware joint trajectory instead of repeatedly solving IK at
    straight-line Cartesian waypoints.
    """
    grasp_sid = site_id(model, "grasp_site")
    start = data.site_xpos[grasp_sid].copy()
    quat = body_quat(model, data, carried_box) if carried_box else None

    for alpha in np.linspace(0.0, 1.0, frames):
        waypoint = (1.0 - alpha) * start + alpha * target_pos
        set_target_marker(data, waypoint)
        solve_ik(model, data, waypoint)
        if carried_box:
            # Manual attachment: keep the box at the gripper site.
            set_free_body_pose(model, data, carried_box, data.site_xpos[grasp_sid].copy(), quat)
        step_viewer(model, data, viewer)


def execute_step(model, data, viewer, step: Step) -> None:
    """Execute one pick-place step using arm IK targets."""
    box_pos = body_pos(model, data, step.source)
    bin_pos = body_pos(model, data, step.target)

    # These are task-space waypoints. The planner decides where the gripper
    # should go; IK decides how the joints should move to get there.
    approach_box = box_pos + np.array([0.0, 0.0, 0.22])
    grasp_box = box_pos + np.array([0.0, 0.0, 0.055])
    lift_box = box_pos + np.array([0.0, 0.0, 0.30])
    approach_bin = bin_pos + np.array([0.0, 0.0, 0.30])
    place_bin = bin_pos + np.array([0.0, 0.0, 0.08])

    print(f"Executing with IK: pick {step.color} -> place in {step.target}")
    move_arm_to(model, data, viewer, approach_box)
    move_arm_to(model, data, viewer, grasp_box, frames=35)
    move_arm_to(model, data, viewer, lift_box, frames=45, carried_box=step.source)
    move_arm_to(model, data, viewer, approach_bin, frames=80, carried_box=step.source)
    move_arm_to(model, data, viewer, place_bin, frames=45, carried_box=step.source)

    final_box_pos = bin_pos + np.array([0.0, 0.0, 0.09])
    set_free_body_pose(model, data, step.source, final_box_pos, body_quat(model, data, step.source))
    move_arm_to(model, data, viewer, approach_bin, frames=35)


def world_state(model, data) -> dict[str, dict[str, list[float]]]:
    """Return the current object state in a digital-twin-friendly format."""
    return {
        box_name: {
            "color": color,
            "position": body_pos(model, data, box_name).round(3).tolist(),
        }
        for color, box_name in BOX_BY_COLOR.items()
    }


def main() -> None:
    """Load the arm scene, build a plan, run IK pick-place, and show it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", default="sort red, blue, green")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    # Compute initial positions for bodies and sites before reading them.
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

        home = np.array([-0.20, 0.0, 0.38])
        move_arm_to(model, data, viewer, home, frames=80)

        for step in plan:
            execute_step(model, data, viewer, step)

        print("Final world state:", world_state(model, data))
        print("Done. Close the viewer window to exit.")
        while viewer.is_running():
            step_viewer(model, data, viewer)


if __name__ == "__main__":
    main()
