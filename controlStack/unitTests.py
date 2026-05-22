if not __debug__:
    from client_v2 import get_clock_box, getCoords, find_item_with_id, is_robot_moving, CameraStream, send_to_VLM,display,robot_controls
    import FSM
    from LowLevelFSM import *
    from ultralytics import YOLO
#We only need to change our Q/A to do unit tests, our robot just listens. Plan is to make a custom GUI to make swapping easier hopefully.
import tkinter as tk

global fsm
global ll_fsm 
global camera
global SERVER_URL

if not __debug__:
    fsm = FSM.SpatialVLMFSM()
    ll_fsm = LowLevelFSM(Point(0,0))
    camera = CameraStream(1).start()
    SERVER_URL = "https://ik92uwhwu2vm2v-8000.proxy.runpod.net" #FIX THIS LATER if we have a dedicated server, probably put this in an env
    model = YOLO('yolo11n.pt')

#5/21 goals: DirectionToClosestBench
# DirectionToClosestStopSign
# AvoidObstacleToReachClosestBench
# AvoidObstacleToReachClosestStopSign
# ClosestBenchWithPerson
class UnitTest:
    def DirectionToClosestBench():
        #Right now I am imagining this as a static test, so lets just clear camera buffer and run through the VLA
        #We are essentially at the start of our original showcase
        fsm.override_states("INIT")
        for _ in range(100):
            camera.read()
        img = camera.read()
        ans = send_to_VLM(img,f"{fsm.question_dict['DirectionToClosestBench']}" )
        return ans.json()['text']
    
    def DirectionToClosestStopSign():
        #Right now I am imagining this as a static test, so lets just clear camera buffer and run through the VLA
        #At this point we should have no people on the board, just the robot and the animals
        fsm.override_states("DRIVETONEARESTSTOP")
        for _ in range(100):
            camera.read()
        img = camera.read()
        ans = send_to_VLM(img,f"{fsm.question_dict['DirectionToClosestStopSign']}")
        return ans.json()['text']


    def AvoidObstacleToReachClosestBench():
        fsm.override_states("DRIVETONEARESTBENCH")
        img = camera.read()

        prompt = f"{fsm.question_dict['ClosestToFurthestBenches']}"
        response = send_to_VLM(img,prompt).json()['text']
        fsm.update_observations({'ClosestToFurthestBenches': response})
        
        #This is our first dynamic test
        img = camera.read()
        results = model(
            source=img, #The camera port
            device="cpu",  #Cuda device (gpu), what is this? LOL we are definetly talking, and processing through the FSM using our cpu, we should probably offload this to the GPU
            #This might've just been an artifact of our testing.
            verbose=False #shuts it up
        )

        print("YOLO LATENCY" + str(time.perf_counter()-start))

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
            results = model(source=img, device="cpu", verbose=False)

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
            direction_prompt = (f"{fsm.question_dict['AvoidObstacleToReachClosestBench']} respond with \'keep straight\', \'go left\', \'go right\' ")
            direction_response = send_to_VLM(img, direction_prompt).json()['text']
            print(f"VLM Direction Response: {direction_response}")
            target_coords = find_item_with_id(results, target, id)
            print(f'robot: {robot_center}, target: {target_coords}')
            item_point = Point(target_coords[0], -1*target_coords[1])
            heading_to_target = robot_center.get_heading(item_point)

            all_commands = robot_controls(direction_response=direction_response)
            
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

            img = camera.read()
            prompt = (f"{fsm.question_dict['ArrivedAtBench']} respond with a \'yes\' or \'no\'")
            exit_answer = send_to_VLM(img,prompt)
        

    def AvoidObstacleToReachClosestStopSign():
        fsm.override_states("DRIVETONEARESTSTOP")
        img = camera.read()
        prompt = f"{fsm.question_dict['ClosestToFurthestStopSigns']}"
        response = send_to_VLM(img,prompt).json()['text']
        fsm.update_observations({'ClosestToFurthestStopSigns': response})
        
        #This is our first dynamic test
        img = camera.read()

        results = model(
            source=img, #The camera port
            device="cpu",  #Cuda device (gpu), what is this? LOL we are definetly talking, and processing through the FSM using our cpu, we should probably offload this to the GPU
            #This might've just been an artifact of our testing.
            verbose=False #shuts it up
        )

        print("YOLO LATENCY" + str(time.perf_counter()-start))

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
            results = model(source=img, device="cpu", verbose=False)

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
            direction_response = send_to_VLM(img, direction_prompt).json()['text']
            print(f"VLM Direction Response: {direction_response}")
            target_coords = find_item_with_id(results, target, id)
            print(f'robot: {robot_center}, target: {target_coords}')
            item_point = Point(target_coords[0], -1*target_coords[1])
            heading_to_target = robot_center.get_heading(item_point)

            all_commands = robot_controls(direction_response=direction_response)
            
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

            img = camera.read()
            prompt = (f"{fsm.question_dict['ArrivedAtAnimalsAroundStopSigns']} respond with a \'yes\' or \'no\'") #Just end it here for the unit test
            exit_answer = send_to_VLM(img,prompt)


    def ClosestBenchWithPerson():
        #Static again
        fsm.override_states("INIT")
        for _ in range(100):
            camera.read()
        img = camera.read()
        ans = send_to_VLM(img,fsm.question_dict["ClosestBenchWithPerson"])
        return ans.json()['text']

