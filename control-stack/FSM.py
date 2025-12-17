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
            'target_bench': -1,
            'at_bench': None,
            'at_stop': None,
            'people_waiting_current_bench': None,
            'target_zoo': -1,
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

        self.question_dict = {
            'people_waiting': 'Are there 1 or more people waiting in this image? Answer Yes or No.',
            'occupied_benches': '',
            'target_bench': '', # shouldn't be asked directly (we keep track of it here)
            'at_bench': f'Is the robot at the bench {self.observations["target_bench"]}? Answer Yes or No.',
            'at_stop': f'Is the robot at the zoo stop {self.observations["target_zoo"]}? Answer Yes or No.',
            'people_waiting_current_bench': 'Are there people at the bench closest to the clock? Respond with Yes or No.',
            'target_zoo': '', # shouldn't be asked directly (we keep track of it here)
            'zoos_to_visit': '', 
            'all_zoos_visited': '', # shouldn't be asked directly (we keep track of it here)
            'waiting_time_exceeded': '', # shouldn't be asked directly (we keep track of it here)
            
        }

    def update_observations(self, vlm_observations):
        for key, value in vlm_observations.items():
            # parse VLM response based on expected type
            value = self.parse_VLM_response(key, value)

            if key in self.observations:
                self.observations[key] = value

        # update observation dependent questions
        self.question_dict['at_bench'] = f'Is the robot at the bench {self.observations["target_bench"]}? Answer Yes or No.'
        self.question_dict['at_stop'] = f'Is the robot at the zoo stop {self.observations["target_zoo"]}? Answer Yes or No.'

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

    def parse_VLM_response(self, obs_key, response):
        response = response.strip().lower()
        if obs_key in ['people_waiting', 'at_bench', 'at_stop', 'people_waiting_current_bench', 'waiting_time_exceeded', 'all_zoos_visited']:
            if response in ['yes', 'true']:
                return True
            elif response in ['no', 'false']:
                return False
        elif obs_key in ['occupied_benches', 'zoos_to_visit']:
            # expecting a list of integers
            try:
                items = response.strip('[]').split(',')
                return [int(item.strip()) for item in items if item.strip().isdigit()]
            except:
                return []
        elif obs_key in ['target_zoo', 'target_bench']: # shouldn't have to parse these directly, but just in case
            try:
                return int(response)
            except:
                return -1
        return response

    def get_current_state(self):
        return self.current_state
    
    def get_relevant_observation_keys(self):
        transition = self.state_transitions.get(self.current_state)
        if transition:
            obs_key, _ = transition
            if isinstance(obs_key, tuple):
                return list(obs_key)
            else:
                keys = [obs_key]
                # in init, we also need to get bench or zoo list based on people_waiting
                if self.current_state == 'INIT':
                    if self.observations['people_waiting']:
                        # also need to get occupied_benches and zoos_to_visit
                        keys.append('occupied_benches')
                    else:
                        keys.append('zoos_to_visit')
                return keys
        return []
    
    def get_relevant_questions(self):
        keys = self.get_relevant_observation_keys()
        return {key: self.question_dict[key] for key in keys if key in self.question_dict}




