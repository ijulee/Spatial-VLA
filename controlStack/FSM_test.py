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
    print(fsm.get_relevant_observation_keys())
    
    observations = {}
    for key in fsm.observations.keys():
        user_input = input(f"Enter value for observation '{key}' (or '' to leave unchanged): ")
        if user_input.lower() != '':
            # Convert input to appropriate type (bool, list, etc.)
            if user_input.lower() in ['true', 'false']:
                observations[key] = user_input.lower() == 'true'
            elif user_input.startswith('[') and user_input.endswith(']'):
                observations[key] = eval(user_input)  # Caution: using eval
            else:
                observations[key] = user_input
    fsm.update_observations(observations)

    # print observations
    print("Updated Observations:")
    for key, value in fsm.observations.items():
        print(f"  {key}: {value}")
    print("------")