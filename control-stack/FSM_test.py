from FSM import SpatialVLMFSM


# create fsm

fsm = SpatialVLMFSM()

# read from terminal input to get user observations
while True:
    print(f"Current State: {fsm.get_current_state()}")
    if fsm.get_current_state() == 'END':
        print("FSM has reached END state. Exiting.")
        break    
    
    print("Relevant Observation Keys for Next Transition:")
    rel_keys = fsm.get_relevant_observation_keys()
    print(rel_keys)

    observations = {}
    for key in rel_keys:
        user_input = input(f"Enter value for observation '{key}' (or '' to leave unchanged): ")
        if user_input != '':
            observations[key] = user_input
    fsm.update_observations(observations)

    # print observations
    print("Updated Observations:")
    for key, value in fsm.observations.items():
        print(f"  {key}: {value}")
    print("target: ", fsm.get_target())
    print("------")