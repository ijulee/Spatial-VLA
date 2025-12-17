from FSM import SpatialVLMFSM
import client as VLM_client
import cv2
from ultralytics import YOLO

if __name__ == "__main__":
    # create fsm
    fsm = SpatialVLMFSM()
    robot_client = ... # robot bluetooth client initialization here

    # camera
    camera_id = 0
    camera = cv2.VideoCapture(camera_id)

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

        # print observations
        print("Updated Observations:")
        for key, value in fsm.observations.items():
            print(f"  {key}: {value}")
        print("------")
