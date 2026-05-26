import base64
from time import time
import tkinter as tk
import inspect
# from unitTests import UnitTest 
# from client_v2 import prefixes_to_remove 
import cv2
#START UP GUI HERE
import json
import json
import numpy as np
import requests
import base64
import io
import threading
import cv2
from PIL import Image
import time
import BluetoothBot
import FSM
from LowLevelFSM import *
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import torch
# from ObjectDetection import ObjectDetection
from ultralytics import YOLO
from client_v2 import get_clock_box, getCoords, find_item_with_id, is_robot_moving, CameraStream,display,robot_controls, send_to_VLM
fsm = FSM.SpatialVLMFSM()
ll_fsm = LowLevelFSM(Point(0,0))
camera = CameraStream(1).start()
robot =  BluetoothBot.BluetoothBot()
robot.open_connection()
# SERVER_URL = "https://ik92uwhwu2vm2v-8000.proxy.runpod.net" #FIX THIS LATER if we have a dedicated server, probably put this in an env
yolo_model = YOLO('yolo11n.pt')

prefixes_to_remove = [
            "ASSISTANT:",
            "Assistant:",
            "assistant:",
            "[INST]",
            "</s>",
            "<s>",
        ]

def start_vla():
    global model, processor
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    print("Loading model...")
    graid_path = "controlStack/best_checkpoint"
    # model = LlavaNextForConditionalGeneration.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.bfloat16,
    #     device_map="cuda",
    #     # token=token  # Pass token explicitly
    # )
    # processor = AutoProcessor.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    # processor = LlavaNextProcessor.from_pretrained(graid_path,use_fast=True)
    processor = AutoProcessor.from_pretrained(
    graid_path,use_fast=True
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, device_map=device, torch_dtype=dtype
    )

    model = PeftModel.from_pretrained(model, graid_path)
    # model = model.merge_and_unload() Optionally merge base model with Lora weights
    model.eval()

def send_to_VLM(img,phase) :
    # img = cv2.imread(img_path_test,cv2.IMREAD_COLOR)
    if img is not None:
        # sucess,buffer = cv2.imencode('.jpg',img)

        # if sucess:
            # img_bytes = buffer.tobytes()
            # img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            # try:
        prompt = phase
        print("RECIEVED IMAGE STARTING INFERENCE")
        # prompt =   "Give me the coordinates of the closest man in the type of a python Dict (ONLY GIVE ME THE DICT) {\"x\": x_val, \"y\": y_val}, if you cant give me {\"x\": x_val, \"y\": y_val} NEVER explain yourself, only the dict" if (phase=="People")  else "Give me the coordinates of the closest Bench in the type of a python Dict (ONLY GIVE ME THE DICT) {'x': x_val, 'y': y_val}, if you cant give me {'x': x_val, 'y': y_val} NEVER explain yourself, only the dict" 
        # payload = {
        # "image": img,
        # "prompt": prompt,
        max_tokens=10000000000 #MESS AROUND WITH THIS
        temperature= 0.5
        # }
        # print(payload["image"])
        # payload_json = json.dumps(payload)
        # print(len(payload_json.encode('utf-8')))
        # start = time.perf_counter()
        # response = requests.post(
        #     f"{SERVER_URL_ALT}/inference",
        #     json=payload
        #     # timeout=30
        # )




    # try:
    start = time.perf_counter() 
    # Decode base64 -> OpenCV -> RGB -> PIL Image
    # img_bgr = base64_to_opencv(payload_json.image)
    # img_rgb = cv2.cvtColor(payload_json, cv2.COLOR_BGR2RGB)
    # print(img_rgb.shape)
    # Convert to PIL Image (what the processor expects)
    from PIL import Image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img)
    

    # Prepare messages for Llama 3.2 Vision
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Apply chat template to get the formatted prompt
    input_text = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Process image and text separately, then combine
    inputs = processor(
        images=[pil_image],
        text=[input_text],
        return_tensors="pt"
    ).to("cuda")
    
    # Move inputs to GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            # stopping_criteria=StoppingCriteria([CancellationCriteria()])
        )
    
    # Decode output
    # generated_text = processor.decode(output[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (remove prompt)
    # The response usually comes after "assistant" or similar marker
    # print(response_text)
    # if "assistant" in generated_text.lower():
        # response_text = generated_text.split("assistant")[-1].strip()
    prompt_ids = processor.tokenizer(input_text, return_tensors="pt")["input_ids"][0]
    # gen_ids = output[0][prompt_ids.shape[0]:]  
    input_len = inputs["input_ids"].shape[1]
    gen_ids = output[0][input_len:]  
    
    # response_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    # response_text = response_text.split("[/INST]")[-1]
    response_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if "[/INST]" in response_text:
        # Everything after [/INST] is the actual answer
        response_text = response_text.split("[/INST]")[-1].strip()
    # else:
    #     # Fallback: just remove the input prompt
    #     response_text = generated_text.replace(input_text, "").strip()
    

    for prefix in prefixes_to_remove:
        if response_text.startswith(prefix):
            response_text = response_text[len(prefix):].strip()

    if "</s>" in response_text:
        response_text = response_text.split("</s>")[0].strip()

    print(time.perf_counter() -start)
    


    response = response_text
    timing = time.perf_counter() - start
    print(response)
    print("RESPONSE TIME: " , timing, " seconds" )
    return (response)
    # except Exception as e:
    #     print(f"Error during VLM inference: {e}")
    #     return e
        

class UnitTest:
    def DirectionToClosestBench():
        #Right now I am imagining this as a static test, so lets just clear camera buffer and run through the VLA
        #We are essentially at the start of our original showcase
        fsm.override_states("INIT")
        # for _ in range(100):
        camera.update()
        camera.read()
        img = camera.read()

        display(img)
        print(f"Sending image to VLM for {fsm.question_dict['DirectionToClosestBench']}...")
        ans = send_to_VLM(img,fsm.question_dict['DirectionToClosestBench'])
        # print("\n" + "ANSWER: " + ans)
        return ans
    
    def DirectionToClosestStopSign():
        #Right now I am imagining this as a static test, so lets just clear camera buffer and run through the VLA
        #At this point we should have no people on the board, just the robot and the animals
        fsm.override_states("DRIVETONEARESTSTOP")
        # for _ in range(100):
        camera.update()
        camera.read()
        img = camera.read()
        ans = send_to_VLM(img,f"{fsm.question_dict['DirectionToClosestStopSign']}")
        return ans


    def AvoidObstacleToReachClosestBench():
        fsm.override_states("DRIVETONEARESTBENCH")
        # for _ in range(100):
        camera.update()
        camera.read()
        img = camera.read()

        prompt = f"{fsm.question_dict['ClosestToFurthestBenches']}"
        response = send_to_VLM(img,prompt)
        fsm.update_observations({'ClosestToFurthestBenches': response})
        
        #This is our first dynamic test
        img = camera.read()
        results = yolo_model(
            source=img, #The camera port
            device="cpu",  #Cuda device (gpu), what is this? LOL we are definetly talking, and processing through the FSM using our cpu, we should probably offload this to the GPU
            #This might've just been an artifact of our testing.
            verbose=False #shuts it up
        )

        # print("YOLO LATENCY" + str(time.perf_counter()-start))

        boxed_img =  results[0].plot()
        display(boxed_img)

        robot_info = get_clock_box(results)
        if(robot_info != None):

            robot_center = Point(float(robot_info['center'][0]), float(-1*robot_info['center'][1]))
            ll_fsm.update_robot_state(robot_center)

            # questions = fsm.get_relevant_questions()


        print('moving forward a little to get header')
        commands = ll_fsm.go_forward(3)
        for c in commands:
            print(f"Sending command: {c}")
            try:
                robot.send_message(c)
            except:
                print("comms eror")
            time.sleep(1) # wait for robot to process command

        exit_answer = ""

        while(exit_answer != "yes"):
            camera.update()
            img = camera.read()
            results = yolo_model(source=img, device="cpu", verbose=False)

            display(results[0].plot())

            robot_info = get_clock_box(results)
            #I do not think heading is a problem anymore(?)
            if(robot_info == None):
                robot_center = Point(0,0)
                ll_fsm.update_robot_state(robot_center)
            else:
                temp_y = -1*float(robot_info['center'][1])
                robot_center = Point(float(robot_info['center'][0]), temp_y)
                ll_fsm.update_robot_state(robot_center)
            direction_prompt = (f"{fsm.question_dict['AvoidObstacleToReachClosestBench']}  ")
            direction_response = send_to_VLM(img, direction_prompt)
            target,id = fsm.get_target()
            print(f"VLM Direction Response: {direction_response}")
            target_coords = find_item_with_id(results, target, id)
            print(f'robot: {robot_center}, target: {target_coords}')
            item_point = Point(target_coords[0], -1*target_coords[1])
            heading_to_target = robot_center.get_heading(item_point)
            
            all_commands = robot_controls(direction_response=direction_response,heading_to_target=heading_to_target,ll_fsm=ll_fsm)
            
            all_commands.append(ll_fsm.go_forward(10))
            # print(f'robot header: {robot_heading}')
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

            img = camera.read()
            prompt = (f"{fsm.question_dict['ArrivedAtBench']} respond with a \'yes\' or \'no\'")
            exit_answer = send_to_VLM(img,prompt)
        

    def AvoidObstacleToReachClosestStopSign():
        fsm.override_states("DRIVETONEARESTSTOP")
        # for _ in range(100):
        camera.update()
        camera.read()
        img = camera.read()
        prompt = f"{fsm.question_dict['ClosestToFurthestStopSigns']}"
        response = send_to_VLM(img,prompt)
        fsm.update_observations({'ClosestToFurthestStopSigns': response})
        
        #This is our first dynamic test
        img = camera.read()

        results = yolo_model(
            source=img, #The camera port
            device="cpu",  #Cuda device (gpu), what is this? LOL we are definetly talking, and processing through the FSM using our cpu, we should probably offload this to the GPU
            #This might've just been an artifact of our testing.
            verbose=False #shuts it up
        )

        # print("YOLO LATENCY" + str(time.perf_counter()-start))

        boxed_img =  results[0].plot()
        display(boxed_img)

        robot_info = get_clock_box(results)
        if(robot_info != None):
            robot_center = Point(float(robot_info['center'][0]), float(-1*robot_info['center'][1]))
            ll_fsm.update_robot_state(robot_center)

            # questions = fsm.get_relevant_questions()


        print('moving forward a little to get header')
        commands = ll_fsm.go_forward(3)
        for c in commands:
            print(f"Sending command: {c}")
            try:
                robot.send_message(c)
            except:
                print("comms eror")
            time.sleep(1) # wait for robot to process command

        exit_answer = ""

        while(exit_answer != "yes"):
            img = camera.read()
            results = yolo_model(source=img, device="cpu", verbose=False)

            display(results[0].plot())

            robot_info = get_clock_box(results)
            #I do not think heading is a problem anymore(?)
            if(robot_info == None):
                robot_center = Point(0,0)
                ll_fsm.update_robot_state(robot_center)
            else:
                temp_y = -1*float(robot_info['center'][1])
                robot_center = Point(float(robot_info['center'][0]), temp_y)
                ll_fsm.update_robot_state(robot_center)
            direction_prompt = (f"{fsm.question_dict['AvoidObstacleToReachClosestStopSign']} respond with \'keep straight\', \'go left\', \'go right\'")
            direction_response = send_to_VLM(img, direction_prompt)
            print(f"VLM Direction Response: {direction_response}")
            target,id = fsm.get_target()

            target_coords = find_item_with_id(results, target, id)
            print(f'robot: {robot_center}, target: {target_coords}')
            item_point = Point(target_coords[0], -1*target_coords[1])
            heading_to_target = robot_center.get_heading(item_point)

            all_commands = robot_controls(direction_response=direction_response, heading_to_target=heading_to_target, ll_fsm=ll_fsm)
            
            all_commands.append(ll_fsm.go_forward(10))
            # print(f'robot header: {robot_heading}')
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

            img = camera.read()
            prompt = (f"{fsm.question_dict['ArrivedAtAnimalsAroundStopSigns']} respond with a \'yes\' or \'no\'") #Just end it here for the unit test
            exit_answer = send_to_VLM(img,prompt)


    def ClosestBenchWithPerson():
        #Static again
        fsm.override_states("INIT")
        for _ in range(100):
            # camera.update()
            camera.read()
        img = camera.read()
        ans = send_to_VLM(img,fsm.question_dict["ClosestBenchWithPerson"])
        return ans


TESTS = [fn for _, fn in inspect.getmembers(UnitTest, predicate=inspect.isfunction)]
def on_run():
    selected_name = selected_var.get()
    # Find the matching function object in your TESTS array
    test_fn = next((f for f in TESTS if f.__name__ == selected_name), None)
    
    if test_fn:

        print(f"Executing: {selected_name}...")
        print(test_fn())
        # for _ in range(10):
        camera.update()
        # camera.read()
        img = camera.read()
        display(img)
        # Optional: Close the GUI automatically after a test finishes running
        # root.destroy() 



if __name__ == "__main__":
    start_vla()
    # camera.start()
    # camera.update()

    # img = camera.read()
    # display(img)
    root = tk.Tk()
    root.title("Spatial-VLA unitTest selector")

    selected_var = tk.StringVar(value=None)

    for fn in TESTS:
        tk.Radiobutton(root, text=fn.__name__, variable=selected_var, value=fn.__name__).pack(anchor="w", padx=10, pady=2)

    tk.Button(root, text="Run Selected", command=on_run).pack(pady=5)
    root.mainloop()

    # while True:
    #     if(cv2.waitKey(0) == 27):  # ESC key
    #         break
    #     img = camera.read()
    #     display(img)
#INFINITE LOOP WAITING ON GUI INPUT

#WE ARE TESTING A ROBOT, SO WE NEED TO DISCARD ALL INPUTS BETWEEN GUI INPUT AND OUTPUT FROM FUNCTION


