
#!/usr/bin/env python3
"""
Quick test - sends a small red square image to your VLM server
"""

import json
import numpy as np
import requests
import base64
import io
import cv2
from PIL import Image
import time
import BluetoothBot
import FSM
from LowLevelFSM import *

toggle = 1
SERVER_URL = "https://ik92uwhwu2vm2v-8000.proxy.runpod.net"
SERVER_URL_ALT = "https://fvb8ktuslhknuz-8000.proxy.runpod.net/" if toggle else "https://ckse5dhl9e73el-8000.proxy.runpod.net/"

img_path_test = "C:/Users/randy/Desktop/Spatial-VLA/Photos/scene_001_bus_06.png"
import cv2 
from ultralytics import YOLO


def send_to_VLM(img,phase) :
    # img = cv2.imread(img_path_test,cv2.IMREAD_COLOR)
    if img is not None:
        sucess,buffer = cv2.imencode('.jpg',img)

        if sucess:
            img_bytes = buffer.tobytes()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            # try:
            prompt = phase
            # prompt =   "Give me the coordinates of the closest man in the type of a python Dict (ONLY GIVE ME THE DICT) {\"x\": x_val, \"y\": y_val}, if you cant give me {\"x\": x_val, \"y\": y_val} NEVER explain yourself, only the dict" if (phase=="People")  else "Give me the coordinates of the closest Bench in the type of a python Dict (ONLY GIVE ME THE DICT) {'x': x_val, 'y': y_val}, if you cant give me {'x': x_val, 'y': y_val} NEVER explain yourself, only the dict" 
            payload = {
                "image": img_b64,
                "prompt": prompt,
                "max_tokens": 10000000000, #MESS AROUND WITH THIS
                "temperature": 0.5
            }
            # print(payload["image"])
            start = time.perf_counter()
            response = requests.post(
                f"{SERVER_URL_ALT}/inference",
                json=payload
                # timeout=30
            )
            timing = time.perf_counter() - start
            # print(response.json())
            print("RESPONSE TIME: " , timing, " seconds" )
        return response
    return None

# def MetaData(results):
    
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             print(box.cid)
def display(img) -> None:
    cv2.imshow("CAMERA VIEW", img) #Just display for live demo purposes i guess
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    


def get_clock_box( results):
    """
    Find the clock box from YOLO results
    
    Returns:
        Dict with clock info or None if not found
    """
    for result in results:
        boxes = result.boxes
        
        # Find clock in detections
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0].cpu().numpy())
            class_name = result.names[class_id]
            
            if class_name == "clock":
                # Get coordinates in different formats
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                xywh = box.xywh[0].cpu().numpy()  # [center_x, center_y, width, height]
                confidence = float(box.conf[0].cpu().numpy())
                
                return {
                    'index': i,
                    'class_name': class_name,
                    'xyxy': xyxy,  # Corners [x1, y1, x2, y2]
                    'xywh': xywh,  # Center format [cx, cy, w, h]
                    'center': ((xywh[0]), (xywh[1])),
                    'confidence': confidence,
                    'bbox': [float(x) for x in xyxy]  # For collision detection
                }
    return None



def getCoords(results,response) -> dict: 
    # return [coords["x"], coords["y"]]
    coords = {"x":0,"y": 0}

    clock_meta_data = get_clock_box(results)
    print("TEST:" ,type(response['text']))
    try:
        coords = json.loads(response['text'])
        print("UNPACK",coords)
        coords = {"x": coords['x']+float(clock_meta_data['xywh'][0]),"y":coords['y']+float(clock_meta_data['xywh'][1])}
        return coords
    except:
        print("missed")
        return None
    
def find_item_with_id(results, item_label, id):
    # from yolo results, find all instances of label. sort from left to right and label, and return the centroid (x,y) with given id (1-indexed)
    try:
        item_coords = []
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0].cpu().numpy())
                label = names[class_id]

                if label == item_label:
                    xywh = box.xywh[0].cpu().numpy()  # [center_x, center_y, width, height]
                    item_coords.append(((xywh[0]).astype(float), (xywh[1]).astype(float)))

        # Sort items from left to right based on x-coordinate
        item_coords.sort(key=lambda coord: coord[0])

        if item_label == 'stop sign':
            return item_coords[0]

        # Return the centroid with the given id (1-indexed)
        if 1 <= id <= len(item_coords):
            return item_coords[id - 1]
        
        else:
            return item_coords[-1]

    except:
        print("error detecting objects")
        return (0,0)

robot_coords = {"x": 0, "y" : 0}

def is_robot_moving(img,supposed_to_move):
    clock_data = get_clock_box(img)
    try:
        if(clock_data == None):
            return False
        if(robot_coords['x'] == clock_data['xywh']['x'] and robot_coords['y'] == clock_data['xywh']['y'] and supposed_to_move):
            return True
        robot_coords['x'] == clock_data['xywh']['x']
        robot_coords['y'] == clock_data['xywh']['y']
    except:
        return False
    return False

def test_bench():
    # img = Image.open("C:/Users/randy/OneDrive/Desktop/Spatial-VLA/Photos/Lab_photo.jpg")
    # img = np.array(img) 

    # # 3. Convert RGB to BGR (Crucial for OpenCV compatibility)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # response = send_to_VLM(img,"People")
    # print(response)
    fsm = FSM.SpatialVLMFSM()
    ll_fsm = LowLevelFSM(Point(0,0))
    # state = local_fsm.get_current_state()
    # print(state)
    camera_id = 1
    robot =  BluetoothBot.BluetoothBot()
    robot.open_connection()
    camera = cv2.VideoCapture(camera_id,cv2.CAP_DSHOW)
    
    camera.read()
    time.sleep(1)
    camera.read()
    model = YOLO('yolo11n.pt')
    first_instance = True
    query=False
    prev_state = ""
    triangulate = False
    heading_data = {"x1": 0, "y1" : 0,  "x2": 0, "y2": 0}
    vlm_observations = {}
    # camera.set(CV_CAP_PROP_BUFFERSIZE, 3)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    # print(fsm.get_relevant_questions())   
    while fsm.get_current_state() != "END":
        start = time.perf_counter()
        for _ in range(700):
            camera.grab()
            # display(img)
        ret,img  = camera.read()
        print(time.perf_counter()-start)


        if(ret == None):
            print("Something went wrong lol")
        else:
            results = model(
                source=img, #The camera port
                device="cpu",  #Cuda device (gpu)
                verbose=False #shuts it up
            )

            boxed_img =  results[0].plot()
            display(boxed_img)

            # get / update robot position
            robot_info = get_clock_box(results)
            if(robot_info != None):

                robot_center = Point(float(robot_info['center'][0]), float(-1*robot_info['center'][1]))
                ll_fsm.update_robot_state(robot_center)

            questions = fsm.get_relevant_questions()

            # if(first_instance or query):
            while(True):
                try:
                    for key, prompt in questions.items():
                        if prompt != '':
                            response = send_to_VLM(img, prompt)
                            print(f"VLM Response for '{prompt}': {response.json()['text']}")
                            vlm_observations[key] = response.json()['text']
                    prev_state = fsm.get_current_state()

                    fsm.update_observations(vlm_observations)
                    print(f"STATE {fsm.get_current_state()}")
                    first_instance=False
                    break
                except KeyboardInterrupt:
                    exit()
                except:
                    print("rerunning")

            # separately handle driving states
            if fsm.get_current_state() in ['DRIVETONEARESTBENCH', 'DRIVETONEARESTSTOP']:
                # on state change, first take a small action and update robot state (to get heading)
                if fsm.get_current_state() != prev_state:
                    print('moving forward a little to get header')
                    commands = ll_fsm.go_forward(3)
                    for c in commands:
                        print(f"Sending command: {c}")
                        try:
                            robot.send_message(c)
                        except:
                            print("comms eror")
                        time.sleep(1) # wait for robot to process command
                    for _ in range(700):
                        camera.grab()
                        # display(img)

                    ret,img = camera.read()
                    results = model(source=img, device="cpu", verbose=False)

                    # cv2.imshow(results[0].plot())
                    display(results[0].plot())

                    robot_info = get_clock_box(results)

                    if(robot_info == None):
                        robot_center = Point(0,0)
                        ll_fsm.update_robot_state(robot_center)
                    else:
                        temp_y = -1*float(robot_info['center'][1])
                        robot_center = Point(float(robot_info['center'][0]), temp_y)
                        ll_fsm.update_robot_state(robot_center)

                
                target,id = fsm.get_target()
                # supposed_to_move = True
                direction_prompt = (f'Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). '
                               f'Use these printed numbers as the stop sign IDs. For {target} number {id}, choose the best '
                               f'direction for the clock to move to reach that {target} while avoiding obstacles between them. '
                               f'Answer with exactly one of: \'keep straight\', \'go left\', \'go right\', \'go up\', or \'go down\'. '
                               f'Answer \'keep straight\' if the clock can move directly toward {target} {id} without colliding with '
                               f'any objects. Answer \'go left\' or \'go right\' if the {target} is mainly above or below the clock and '
                               f'it is better for the clock to pass the nearest obstacle between them on its left or right side. Answer '
                               f'\'go up\' or \'go down\' if the {target} is mainly to the left or right of the clock and it is better '
                               f'for the clock to pass the nearest obstacle between them above or below it.')
                
                # requery with direction prompt:
                direction_response = send_to_VLM(img, direction_prompt).json()['text']
                print(f"VLM Direction Response: {direction_response}")
                target_coords = find_item_with_id(results, target, id)
                print(f'robot: {robot_center}, target: {target_coords}')
                item_point = Point(target_coords[0], -1*target_coords[1])
                heading_to_target = robot_center.get_heading(item_point)

                # choose next action based on VLM response
                all_commands = []
                robot_heading = ll_fsm.robot_state.cur_heading
                if direction_response == 'keep straight':
                    # align heading and go forward
                    all_commands.append(ll_fsm.turn_to_heading(heading_to_target))
                elif direction_response == 'go left':
                    if 0 < robot_heading <= 180: # facing up
                        all_commands.append(ll_fsm.turn_to_heading(heading_to_target + 30)) # CCW
                    else:
                        all_commands.append(ll_fsm.turn_to_heading(heading_to_target - 30)) # CW
                elif direction_response == 'go right':
                    if 0 < robot_heading <= 180: # facing up
                        all_commands.append(ll_fsm.turn_to_heading(heading_to_target - 30)) # CW
                    else:
                        all_commands.append(ll_fsm.turn_to_heading(heading_to_target + 30)) # CCW
                elif direction_response == 'go up':
                    if 0 <= robot_heading <= 90 or 270 < robot_heading <= 360: # facing right
                        all_commands.append(ll_fsm.turn_to_heading(heading_to_target + 30)) # CCW
                    else:
                        all_commands.append(ll_fsm.turn_to_heading(heading_to_target - 30)) # CW
                elif direction_response == 'go down':
                    if 0 <= robot_heading <= 90 or 270 < robot_heading <= 360: # facing right
                        all_commands.append(ll_fsm.turn_to_heading(heading_to_target - 30)) # CW
                    else:
                        all_commands.append(ll_fsm.turn_to_heading(heading_to_target + 30)) # CCW

                all_commands.append(ll_fsm.go_forward(10))
                print(f'robot header: {robot_heading}')
                print(f'header to object: {heading_to_target}')
                # send commands to robot
                for commands in all_commands:
                    for command in commands:
                        print(f"Sending command: {command}")
                        try:
                            robot.send_message(command)
                        except:
                            print("comms error")
                        time.sleep(2) # wait for robot to process command
                
            elif fsm.get_current_state() == 'VIEWANIMALS':
                # block for 5 seconds and then add observation
                print('viewing animals...')
                time.sleep(5)
                observation = {'waiting_time_exceeded': 'True'}
                fsm.update_observations(observation)
                print('done viewing animals.')
                
            # only query again once robot stops moving
            # query = is_robot_moving(results, supposed_to_move) or not supposed_to_move
        

if __name__ == "__main__":
    # project_pipeline()
    test_bench()
            
    cv2.destroyAllWindows()