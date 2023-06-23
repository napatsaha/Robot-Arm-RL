# Robot-Arm-RL
 WSU Postgraduate Project Project A &amp; B    Reinforcement Learning with Robot Arm

This github directory contains both the Python files to train the RL algorithms (Training\) and the Unity projects where different variations of the environments are stored.

## Unity Project Structure

#### Example

<!-- <figure class="video_container">
  <video controls="true" allowfullscreen="true">
    <source src="docs/videos/robot_subset_joint2_102.mp4" type="video/mp4">
  </video>
</figure> -->



https://github.com/WSU-Data-Science/Robot-Arm-RL/assets/98146048/c8fd5824-a286-4048-88e0-56d917e90983



#### Different Projects

There are so far 4 variations of the Unity project environments used for training the robot (in chronological order of development):

- **ArmRobotOriginal** - The original environment with:
  - all 7 joints fully operational
  - Discrete actions (0,1,2)
  - The target spawning space as a sphere around the base
- **ArmRobotSubset** - Modified version (corresponding to "robot_subset.py"):
	- Can control where target spawn depending on which joint is active
	- Discrete actions (0,1,2) for 7 joints
- **ArmRobotContinuous** - Similar to *ArmRobotSubset* but with continuous actions:
	- Continuous actions [-1,1] for all 7 joints
	- Target spawn controllable by which joint is active
- **ArmRobotExpanded** - Redefined reward function and observation space (still in development)

#### Joint-Dependent Target Position Spawning
For environments that accept active joints to spawn target, the target spawns in a circle surrounding the center position of the *active joint*. 
The orientation of the circle depends on the axis of rotation of the joint (for example, if the *active joint* rotates horizontally -- around the y-axis -- the circle is orientated in a horizontal plane).

The *active joint* is defined as the lowest joint that is operational. Any joint above that joint is assumed to be operational, while lower joints are assumed fixed. Joints are ordered from the base to the tip, with joint `0` being the bottom-most joint, and joint `6` being the tip joint (we assume this joint is useless because it only rotates the pincher).



#### Manual Controls

You can move the robot around manually using the following keyboard commands:

```
A/D - rotate base joint
S/W - rotate shoulder joint
Q/E - rotate elbow joint
O/P - rotate wrist1
K/L - rotate wrist2
N/M - rotate wrist3
V/B - rotate hand
X - close pincher
Z - open pincher
```

This is controlled in the `Heuristic` method of the `RobotAgent.cs` script of each environment


## Presentation Slides

[Can be found here...](napatsaha.github.io/PPB2023/)

