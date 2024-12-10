from statemachine import StateMachine, State
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

# Define the FSM


class SLAMFSM(StateMachine):
    idle = State("Idle", initial=True)
    decide_move = State("Decide Move")
    exploration = State("Exploration")
    detection = State("Dynamic Object Detection")
    mapping = State("Mapping")
    localization = State("Localization")
    loop_closure = State("Loop Closure")
    complete = State("Complete", final=True)

    start = idle.to(decide_move)
    decide_direction = decide_move.to(exploration)
    detect_objects = exploration.to(detection)
    update_map = detection.to(mapping)
    localize = mapping.to(localization)
    close_loop = localization.to(loop_closure)
    continue_explore = loop_closure.to(decide_move)
    finish = loop_closure.to(complete)


# Initialize FSM
slam_fsm = SLAMFSM()

# Environment parameters
GRID_SIZE = 16
environment = np.full((GRID_SIZE, GRID_SIZE), -1)  # -1 indicates unexplored
bot_position = [GRID_SIZE // 2, GRID_SIZE // 2]  # Start in the center
fov_radius = 3

# Function to generate static objects


def generate_static_objects(grid_size, num_objects=20):
    static_objects = []
    for _ in range(num_objects):
        shape_type = random.choice(["point", "line", "square", "angle"])
        x, y = random.randint(
            0, grid_size - 1), random.randint(0, grid_size - 1)

        if shape_type == "point":
            static_objects.append((x, y))
        elif shape_type == "line":
            for i in range(3):
                if random.choice([True, False]):
                    if y + i < grid_size:
                        static_objects.append((x, y + i))
                else:
                    if x + i < grid_size:
                        static_objects.append((x + i, y))
        elif shape_type == "square":
            for i in range(3):
                for j in range(3):
                    if x + i < grid_size and y + j < grid_size:
                        static_objects.append((x + i, y + j))
        elif shape_type == "angle":
            for i in range(3):
                if x + i < grid_size:
                    static_objects.append((x + i, y))
                if y + i < grid_size:
                    static_objects.append((x, y + i))

    return list(set(static_objects))


# Generate static and dynamic objects
static_objects = generate_static_objects(GRID_SIZE)
dynamic_objects = [(random.randint(0, GRID_SIZE - 1),
                    random.randint(0, GRID_SIZE - 1)) for _ in range(2)]
dynamic_objects = [(x, y)
                   for (x, y) in dynamic_objects if (x, y) not in static_objects]

# Precompute FSM graph layout
fsm_graph = nx.DiGraph()
states = [
    "Idle", "Decide Move", "Exploration", "Dynamic Object Detection",
    "Mapping", "Localization", "Loop Closure", "Complete"
]
transitions = [
    ("Idle", "Decide Move"), ("Decide Move", "Exploration"),
    ("Exploration", "Dynamic Object Detection"),
    ("Dynamic Object Detection", "Mapping"), ("Mapping", "Localization"),
    ("Localization", "Loop Closure"), ("Loop Closure", "Decide Move"),
    ("Loop Closure", "Complete")
]
fsm_graph.add_nodes_from(states)
fsm_graph.add_edges_from(transitions)
fsm_pos = nx.spring_layout(fsm_graph)

# FSM graph visualization


def update_fsm_graph(current_state):
    node_colors = ["red" if state ==
                   current_state else "lightblue" for state in states]
    nx.draw(fsm_graph, pos=fsm_pos, with_labels=True,
            node_color=node_colors, node_size=1500, font_size=10, arrows=True)
    plt.title("FSM Visualization")

# Visualization function


def visualize_environment(state_description):
    plt.clf()

    # Environment visualization
    plt.subplot(1, 2, 1)
    plt.imshow(environment, cmap="viridis", origin="upper", vmin=-1, vmax=1)

    # Highlight bot position
    plt.scatter(bot_position[1], bot_position[0], color="green",
                label=f"Bot: {bot_position}", s=100, edgecolors="black")

    # Annotate objects
    for obj in dynamic_objects:
        plt.text(obj[1], obj[0], "D", color="red",
                 ha="center", va="center", fontsize=10)
    for obj in static_objects:
        plt.text(obj[1], obj[0], "S", color="blue",
                 ha="center", va="center", fontsize=10)

    # Add title and legend
    plt.title(
        f"Current State: {slam_fsm.current_state.id}\n{state_description}", fontsize=12)
    plt.xlabel("Legend: D = Dynamic Object, S = Static Object, Green = Bot")
    plt.colorbar(label="Occupancy Value")
    plt.grid(True)
    plt.legend()

    # FSM visualization
    plt.subplot(1, 2, 2)
    update_fsm_graph(slam_fsm.current_state.id)
    plt.tight_layout()
    plt.pause(0.5)

# Move bot with collision avoidance


def move_bot():
    for _ in range(10):  # Try 10 times to find a valid position
        new_x = max(
            0, min(GRID_SIZE - 1, bot_position[0] + random.choice([-1, 0, 1])))
        new_y = max(
            0, min(GRID_SIZE - 1, bot_position[1] + random.choice([-1, 0, 1])))
        if (new_x, new_y) not in static_objects and (new_x, new_y) not in dynamic_objects:
            bot_position[0], bot_position[1] = new_x, new_y
            break

# Move dynamic objects


def move_dynamic_objects():
    for i, (x, y) in enumerate(dynamic_objects):
        new_x = max(0, min(GRID_SIZE - 1, x + random.choice([-1, 0, 1])))
        new_y = max(0, min(GRID_SIZE - 1, y + random.choice([-1, 0, 1])))
        if (new_x, new_y) not in static_objects and (new_x, new_y) != bot_position:
            dynamic_objects[i] = (new_x, new_y)

# Update environment


def update_environment():
    for x in range(environment.shape[0]):
        for y in range(environment.shape[1]):
            if np.linalg.norm([x - bot_position[0], y - bot_position[1]]) <= fov_radius:
                if (x, y) in dynamic_objects:
                    environment[x, y] = 0.5
                elif (x, y) in static_objects:
                    environment[x, y] = 1.0
                else:
                    environment[x, y] = 0.0


# Main loop
slam_fsm.start()
while slam_fsm.current_state != slam_fsm.complete:
    if slam_fsm.current_state == slam_fsm.decide_move:
        visualize_environment("Deciding next direction...")
        slam_fsm.decide_direction()
    elif slam_fsm.current_state == slam_fsm.exploration:
        move_bot()
        update_environment()
        visualize_environment("Exploring the environment...")
        slam_fsm.detect_objects()
    elif slam_fsm.current_state == slam_fsm.detection:
        move_dynamic_objects()
        update_environment()
        visualize_environment("Detecting dynamic objects...")
        slam_fsm.update_map()
    elif slam_fsm.current_state == slam_fsm.mapping:
        visualize_environment("Updating the map...")
        slam_fsm.localize()
    elif slam_fsm.current_state == slam_fsm.localization:
        visualize_environment("Localizing the robot...")
        slam_fsm.close_loop()
    elif slam_fsm.current_state == slam_fsm.loop_closure:
        visualize_environment("Performing loop closure...")
        if np.all(environment[environment >= 0] == 1.0):
            slam_fsm.finish()
        else:
            slam_fsm.continue_explore()

print("SLAM process complete!")
plt.show()
