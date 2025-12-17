from FSM import SpatialVLMFSM
import client as VLM_client
import cv2
import time
from ultralytics import YOLO

if __name__ == "__main__":
    # create fsm
    fsm = SpatialVLMFSM()
    robot_client = ... # robot bluetooth client initialization here

    # camera
    camera_id = 0
    camera = cv2.VideoCapture(camera_id)

    # before looping, we need to do some initial queries
    img = camera.read()[1]
    questions = fsm.get_relevant_questions(fsm.get_init_keys())
    for key, prompt in questions.items():
        response = VLM_client.send_to_VLM(img, prompt)
        print(f"Initial VLM Response for '{prompt}': {response}")
        fsm.update_observations({key: response})

    # main loop to update FSM based on robot observations
    while True:
        print(f"Current State: {fsm.get_current_state()}")
        if fsm.get_current_state() == 'END':
            print("FSM has reached END state. Exiting.")
            break    

        # check relevant observation keys and get associated questions
        relevant_questions = fsm.get_relevant_questions()
        print("Relevant Observation Keys for Next Transition:")
        for key in relevant_questions:
            print(f"  {key}: {relevant_questions[key]}")

        # get camera img
        img = camera.read()[1]

        # loop through relevant questions and get observations from VLM
        vlm_observations = {}
        for key, prompt in relevant_questions.items():
            response = VLM_client.send_to_VLM(img, prompt)
            print(f"VLM Response for '{prompt}': {response}")
            vlm_observations[key] = response

        # update observations based on responses
        fsm.update_observations(vlm_observations)

        # if driving state, we'll need to requery to get a direction
        if fsm.get_current_state() in ['DRIVETONEARESTBENCH', 'DRIVETONEARESTSTOP']:
            target,id = fsm.get_target()
            direction_prompt = f'... go to {target} {id}'
            direction_response = VLM_client.send_to_VLM(img, direction_prompt)
            print(f"VLM Direction Response: {direction_response}")

            # send command to robot via bluetooth
            if robot_client is not None:
                robot_client.send_command(direction_response)

        # print observations
        print("Updated Observations:")
        for key, value in fsm.observations.items():
            print(f"  {key}: {value}")
        print("------")
