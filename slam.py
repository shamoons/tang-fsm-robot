from statemachine import StateMachine, State
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import random

# Define the FSM


class SLAMFSM(StateMachine):
    idle = State('Idle', initial=True)
    exploration = State('Exploration')
    detection = State('Dynamic Object Detection')
    mapping = State('Mapping')
    localization = State('Localization')
    loop_closure = State('Loop Closure')
    complete = State('Complete')  # New state to signify FSM completion

    is_updated = False

    start = idle.to(exploration)
    detect_objects = exploration.to(detection)
    update_map = detection.to(mapping)
    localize = mapping.to(localization)
    close_loop = localization.to(loop_closure)
    continue_explore = loop_closure.to(exploration)
    finish = loop_closure.to(complete)  # Transition to complete state


# Initialize FSM
slam_fsm = SLAMFSM()

# Simulate Environment
environment = np.zeros((10, 10))  # 10x10 grid
dynamic_objects = [(3, 3), (6, 6)]  # Example positions of dynamic objects
static_objects = [(2, 2), (8, 8)]  # Example positions of static objects
bot_position = [5, 5]  # Starting position of the bot (center of grid)
fov_radius = 2  # Field of view radius (circular region)

# Precompute the FSM graph layout
fsm_graph = nx.DiGraph()
states = [
    "Idle", "Exploration", "Dynamic Object Detection",
    "Mapping", "Localization", "Loop Closure", "Complete"
]
transitions = [
    ("Idle", "Exploration"), ("Exploration", "Dynamic Object Detection"),
    ("Dynamic Object Detection", "Mapping"), ("Mapping", "Localization"),
    ("Localization", "Loop Closure"), ("Loop Closure", "Exploration"),
    ("Loop Closure", "Complete")
]
fsm_graph.add_nodes_from(states)
fsm_graph.add_edges_from(transitions)
fsm_pos = nx.spring_layout(fsm_graph)  # Precompute layout for static graph

# Update FSM graph visualization


def update_fsm_graph(current_state):
    node_colors = ["red" if state ==
                   current_state else "lightblue" for state in fsm_graph.nodes]
    nx.draw(fsm_graph, pos=fsm_pos, with_labels=True,
            node_color=node_colors, node_size=1500, font_size=10, arrows=True)
    plt.title("FSM Visualization")

# Visualization function


def visualize_environment(state_description):
    plt.clf()  # Clear the plot for updating

    # Environment visualization
    plt.subplot(1, 2, 1)
    plt.imshow(environment, cmap='Greys', origin='upper')

    # Highlight the bot's position
    plt.scatter(bot_position[1], bot_position[0],
                color='green', label='Bot', s=100, edgecolors='black')

    # Highlight the bot's FOV
    for x in range(environment.shape[0]):
        for y in range(environment.shape[1]):
            if np.linalg.norm([x - bot_position[0], y - bot_position[1]]) <= fov_radius:
                plt.gca().add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1,
                                                  1, fill=False, edgecolor='yellow', linewidth=1))

    # Annotate objects
    for obj in dynamic_objects:
        plt.text(obj[1], obj[0], 'D', color='red',
                 ha='center', va='center', fontsize=10)
        if np.linalg.norm([obj[0] - bot_position[0], obj[1] - bot_position[1]]) <= fov_radius:
            plt.gca().add_patch(plt.Rectangle(
                (obj[1] - 0.5, obj[0] - 0.5), 1, 1, fill=False, edgecolor='green', linewidth=2))

    for obj in static_objects:
        plt.text(obj[1], obj[0], 'S', color='blue',
                 ha='center', va='center', fontsize=10)
        if np.linalg.norm([obj[0] - bot_position[0], obj[1] - bot_position[1]]) <= fov_radius:
            plt.gca().add_patch(plt.Rectangle(
                (obj[1] - 0.5, obj[0] - 0.5), 1, 1, fill=False, edgecolor='green', linewidth=2))

    # Add title and legend
    plt.title(
        f"Current State: {slam_fsm.current_state.id}\n{state_description}", fontsize=12)
    plt.xlabel(
        "Legend: D = Dynamic Object, S = Static Object, Green = Bot, Yellow = FOV")
    plt.colorbar(label="Occupancy Value")  # Add colorbar for reference
    plt.grid(True)
    plt.legend()

    # FSM visualization
    plt.subplot(1, 2, 2)

    update_fsm_graph(slam_fsm.current_state.id)
    plt.tight_layout()
    plt.pause(0.5)

# Move dynamic objects


def move_dynamic_objects():
    for i, (x, y) in enumerate(dynamic_objects):
        new_x = max(0, min(9, x + random.choice([-1, 0, 1])))
        new_y = max(0, min(9, y + random.choice([-1, 0, 1])))
        dynamic_objects[i] = (new_x, new_y)

# Move the bot


def move_bot():
    bot_position[0] = max(
        0, min(9, bot_position[0] + random.choice([-1, 0, 1])))
    bot_position[1] = max(
        0, min(9, bot_position[1] + random.choice([-1, 0, 1])))


# Main loop
slam_fsm.start()
while slam_fsm.current_state != slam_fsm.complete:
    if slam_fsm.current_state == slam_fsm.exploration:
        move_bot()
        visualize_environment("Exploring the environment to detect objects.")
        slam_fsm.detect_objects()
    elif slam_fsm.current_state == slam_fsm.detection:
        move_dynamic_objects()
        visualize_environment("Detecting and marking dynamic objects.")
        slam_fsm.update_map()
    elif slam_fsm.current_state == slam_fsm.mapping:
        visualize_environment("Updating the map with static objects.")
        slam_fsm.localize()
    elif slam_fsm.current_state == slam_fsm.localization:
        visualize_environment("Localizing the robot within the updated map.")
        slam_fsm.close_loop()
    elif slam_fsm.current_state == slam_fsm.loop_closure:
        visualize_environment("Recognizing and correcting drift in the map.")
        if np.all(environment[environment > 0] == 1.0):  # All objects processed
            slam_fsm.finish()
        else:
            slam_fsm.continue_explore()

print("SLAM process complete!")
plt.show()
