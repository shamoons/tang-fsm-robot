from statemachine import State, StateMachine


class TileLayingRobotFSM(StateMachine):
    # Define states
    idle = State("Idle", initial=True)
    navigate_to_tile_stack = State("Navigate to Tile Stack")
    pick_up_tile = State("Pick Up Tile")
    navigate_to_placement_position = State("Navigate to Placement Position")
    placement_preparation = State("Placement Preparation")
    place_tile = State("Place Tile")
    cycle_reset = State("Cycle Reset")

    # Define transitions
    start_navigation = idle.to(navigate_to_tile_stack)
    start_pick_up = navigate_to_tile_stack.to(pick_up_tile)
    start_placement_navigation = pick_up_tile.to(
        navigate_to_placement_position)
    start_preparation = navigate_to_placement_position.to(
        placement_preparation)
    start_tile_placement = placement_preparation.to(place_tile)
    reset_cycle = place_tile.to(cycle_reset)
    back_to_idle = cycle_reset.to(idle)

    # Actions for each state
    def on_enter_navigate_to_tile_stack(self):
        print("Navigating to the tile stack...")

    def on_enter_pick_up_tile(self):
        print("Picking up a tile...")
        self.simulate_action("Tile picked up successfully!")

    def on_enter_navigate_to_placement_position(self):
        print("Navigating to the placement position...")

    def on_enter_placement_preparation(self):
        print("Preparing to place the tile...")

    def on_enter_place_tile(self):
        print("Placing the tile...")
        self.simulate_action("Tile placed successfully!")

    def on_enter_cycle_reset(self):
        print("Resetting for the next cycle...")

    def simulate_action(self, message):
        # Simulate a delay or success condition
        print(message)


if __name__ == "__main__":
    # Initialize FSM
    robot_fsm = TileLayingRobotFSM()

    # Simulate the process
    print(f"Current State: {robot_fsm.current_state}")
    robot_fsm.start_navigation()
    print(f"Current State: {robot_fsm.current_state}")
    robot_fsm.start_pick_up()
    print(f"Current State: {robot_fsm.current_state}")
    robot_fsm.start_placement_navigation()
    print(f"Current State: {robot_fsm.current_state}")
    robot_fsm.start_preparation()
    print(f"Current State: {robot_fsm.current_state}")
    robot_fsm.start_tile_placement()
    print(f"Current State: {robot_fsm.current_state}")
    robot_fsm.reset_cycle()
    print(f"Current State: {robot_fsm.current_state}")
    robot_fsm.back_to_idle()
    print(f"Current State: {robot_fsm.current_state}")
