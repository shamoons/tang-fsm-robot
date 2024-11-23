import time
import numpy as np
import skrobot

# Initialize the viewer
viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))

# Load a robot model (e.g., KUKA robot)
robot = skrobot.models.Kuka()
viewer.add(robot)

# Define obstacles (e.g., kitchen counters)
obstacles = []

# Countertop 1
counter1 = skrobot.model.Box(extents=[0.5, 1.0, 0.9], with_sdf=True)
counter1.translate([0.75, 0.0, 0.45])
obstacles.append(counter1)

# Countertop 2
counter2 = skrobot.model.Box(extents=[0.5, 1.0, 0.9], with_sdf=True)
counter2.translate([-0.75, 0.0, 0.45])
obstacles.append(counter2)

# Add obstacles to the viewer
for obs in obstacles:
    viewer.add(obs)

# Set the initial position of the robot
robot.translate([0.0, -1.0, 0.0])
viewer.show()

# Define waypoints for navigation
waypoints = [
    [0.0, -0.5, 0.0],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [-0.5, 0.0, 0.0],
    [0.0, -0.5, 0.0]
]

# Function to move the robot to a target position


def move_robot_to(target_pos, steps=100):
    current_pos = robot.translation
    for i in range(steps):
        alpha = (i + 1) / steps
        new_pos = (1 - alpha) * current_pos + alpha * target_pos
        robot.translate(new_pos - robot.translation)
        viewer.redraw()
        time.sleep(0.01)


# Navigate through waypoints
for waypoint in waypoints:
    move_robot_to(np.array(waypoint))
    time.sleep(1.0)

# Keep the viewer open
print('Press [q] to close the window.')
while not viewer.has_exit:
    time.sleep(0.1)
    viewer.redraw()
