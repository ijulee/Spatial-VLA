'''
This module defines a custom FSM class for Group 15 Spatial VLM project.
'''

class SpatialVLMFSM:
    def __init__(self):
        self.states = ['INIT', 'DRIVETONEARESTBENCH', 'PICKUP', 'DRIVETONEARESTSTOP',
                  'VIEWANIMALS', 'END']
        self.current_state = 'INIT' # initial state
        self.observations = {
            'people_waiting': None,
            'occupied_benches': [],
            'target_bench': None,
            'at_bench': None,
            'at_stop': None,
            'people_waiting_current_bench': None,
            'target_zoo': None,
            'zoos_to_visit': [],
            'all_zoos_visited': None,
            'waiting_time_exceeded': None,
        }
        # in state transitions, include guard conditions as needed
        self.state_transitions = {
            'INIT': ('people_waiting', {True: 'DRIVETONEARESTBENCH', False: 'DRIVETONEARESTSTOP'}),
            'DRIVETONEARESTBENCH': ('at_bench', {True: 'PICKUP', False: 'DRIVETONEARESTBENCH'}),
            'PICKUP': ('people_waiting_current_bench', {True: 'PICKUP', False: 'INIT'}),
            'DRIVETONEARESTSTOP': ('at_stop', {True: 'VIEWANIMALS', False: 'DRIVETONEARESTSTOP'}),
            'VIEWANIMALS': (('waiting_time_exceeded', 'all_zoos_visited'), 
                            {(True, True): 'END', (False, True): 'VIEWANIMALS', 
                             (False, False): 'VIEWANIMALS', (True, False): 'DRIVETONEARESTSTOP'}),
        }

    def update_observations(self, observations):
        for key, value in observations.items():
            if key in self.observations:
                self.observations[key] = value

        # after updating observations, check for possible state transition
        self._do_transition()

    def _do_transition(self):
        prev_state = self.current_state
        transition = self.state_transitions.get(self.current_state)
        if transition:
            obs_key, conditions = transition
            if isinstance(obs_key, tuple):
                obs_values = tuple(self.observations[key] for key in obs_key)
                new_state = conditions.get(obs_values)
            else:
                obs_value = self.observations[obs_key]
                new_state = conditions.get(obs_value)

            if new_state and new_state in self.states:
                self.current_state = new_state

        # we can do some manual observation updates
        if prev_state == 'DRIVETONEARESTBENCH' and self.current_state == 'PICKUP':
            self.observations['occupied_benches'].pop(0)  # remove the bench we just went to
            if len(self.observations['occupied_benches']) == 0:
                self.observations['people_waiting'] = False
        elif prev_state == 'VIEWANIMALS' and self.current_state == 'DRIVETONEARESTSTOP':
            self.observations['zoos_to_visit'].pop(0)  # remove the zoo we just visited
            if len(self.observations['zoos_to_visit']) == 0:
                self.observations['all_zoos_visited'] = True
            
        # some observations can be derived from others
        self.observations['target_bench'] = self.observations['occupied_benches'][0] if self.observations['occupied_benches'] else None
        self.observations['target_zoo'] = self.observations['zoos_to_visit'][0] if self.observations['zoos_to_visit'] else None
        self.observations['all_zoos_visited'] = len(self.observations['zoos_to_visit']) == 0

    def get_current_state(self):
        return self.current_state
    
    def get_relevant_observation_keys(self):
        transition = self.state_transitions.get(self.current_state)
        if transition:
            obs_key, _ = transition
            if isinstance(obs_key, tuple):
                return list(obs_key)
            else:
                return [obs_key]
        return []





