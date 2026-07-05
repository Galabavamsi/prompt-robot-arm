# Prompt Robot Arm

Prompt Robot Arm is a simulation-first robotics learning project for exploring
language-conditioned sorting, world state, inverse kinematics, and eventually
physical AI / world-model planning.

The starting task is simple:

> Sort colored boxes into bins according to a prompt such as
> `sort red, blue, green`.

The project is intentionally built in versions. Each version teaches one new
robotics concept while keeping the rest understandable.

## Current Versions

### Version 1: Fake Gripper Sorting

File:

```text
version_1/fake_gripper_sorting.py
```

Scene:

```text
sim/sorting_scene.xml
```

This version uses a kinematic fake gripper. It does not solve robot joint
motion yet. Its purpose is to teach the first robotics intelligence loop:

```text
prompt -> structured plan -> world state -> pick-place sequence
```

What it teaches:

- How a prompt becomes a structured robot plan.
- How simulator state can act like a tiny digital twin.
- How pick-place skills are organized as waypoints.
- Why robots should execute structured actions, not raw language.

### Version 2: Simple Arm With IK

File:

```text
version_2/arm_ik_sorting.py
```

Scene:

```text
sim/sorting_arm_scene.xml
```

This version replaces the fake gripper with a simple 4-DOF arm. It uses
position-only inverse kinematics to move the gripper site to each target point.

What it teaches:

- End effector / tool-center-point thinking.
- Forward kinematics vs inverse kinematics.
- MuJoCo bodies, joints, sites, and free bodies.
- Jacobian-based damped least-squares IK.
- The difference between task planning and motion control.

## Setup

This project was tested from Ubuntu/WSL with Python virtual environments.

Create and activate an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you already have a robotics environment, just install:

```bash
pip install mujoco numpy
```

## Run Version 1

```bash
python version_1/fake_gripper_sorting.py --instruction "sort red, blue, green"
```

Include yellow if you want the yellow box to move too:

```bash
python version_1/fake_gripper_sorting.py --instruction "sort red, blue, green, yellow"
```

Try changing the sequence:

```bash
python version_1/fake_gripper_sorting.py --instruction "sort yellow, red, blue"
```

## Run Version 2

```bash
python version_2/arm_ik_sorting.py --instruction "sort red, blue, green"
```

Try:

```bash
python version_2/arm_ik_sorting.py --instruction "sort yellow, red, blue"
```

## Architecture Idea

The long-term architecture is:

```text
user prompt
  -> language/task planner
  -> structured task plan
  -> world state / digital twin
  -> skill executor
  -> IK / motion planning
  -> robot controller
```

Later versions can replace individual modules:

- Replace simple parsing with an SLM/LLM planner.
- Replace simulator ground-truth state with camera perception.
- Replace direct IK stepping with ROS 2 + MoveIt 2.
- Add a world model to predict slip, collision, instability, or task failure.
- Add predictive maintenance from joint current, temperature, cycle time, and
  gripper-force telemetry.

## Roadmap

- [x] Version 1: fake gripper sorting.
- [x] Version 2: simple arm with inverse kinematics.
- [ ] Version 3: real robot arm model in MuJoCo.
- [ ] Version 4: collision-aware motion planning.
- [ ] Version 5: camera perception instead of simulator ground truth.
- [ ] Version 6: ROS 2 / MoveIt 2 integration.
- [ ] Version 7: Isaac Sim digital twin.
- [ ] Version 8: world-model planning and predictive maintenance.

## Research Direction

This project is a small testbed for physical AI:

> A robot system that reasons over prompts, world state, sensor readings,
> metadata, and predicted future outcomes, while verified robotics software
> handles motion, collision checking, and safety.

The colored-box sorting task is deliberately simple. The real goal is to build
the layers needed for more general robot reasoning.
