
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


SERVER_URL = "https://ik92uwhwu2vm2v-8000.proxy.runpod.net"
SERVER_URL_ALT = "https://ckse5dhl9e73el-8000.proxy.runpod.net/"
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
            prompt = "Count the number of men"
            # prompt =   "Give me the coordinates of the closest man in the type of a python Dict (ONLY GIVE ME THE DICT) {\"x\": x_val, \"y\": y_val}, if you cant give me {\"x\": x_val, \"y\": y_val} NEVER explain yourself, only the dict" if (phase=="People")  else "Give me the coordinates of the closest Bench in the type of a python Dict (ONLY GIVE ME THE DICT) {'x': x_val, 'y': y_val}, if you cant give me {'x': x_val, 'y': y_val} NEVER explain yourself, only the dict" 
            payload = {
                "image": img_b64,
                "prompt": prompt,
                "max_tokens": 100000, #MESS AROUND WITH THIS
                "temperature": 1.0
            }
            # print(payload["image"])
            start = time.perf_counter()
            response = requests.post(
                f"{SERVER_URL_ALT}/inference",
                json=payload
                # timeout=30
            )
            timing = time.perf_counter() - start
            print(response.json())
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
            
            if class_name == "person":
                # Get coordinates in different formats
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                xywh = box.xywh[0].cpu().numpy()  # [center_x, center_y, width, height]
                confidence = float(box.conf[0].cpu().numpy())
                
                return {
                    'index': i,
                    'class_name': class_name,
                    'xyxy': xyxy,  # Corners [x1, y1, x2, y2]
                    'xywh': xywh,  # Center format [cx, cy, w, h]
                    'center': (float(xywh[0]), float(xywh[1])),
                    'confidence': confidence,
                    'bbox': [float(x) for x in xyxy]  # For collision detection
                }
    
    return None


def getPhases(latch, results):
    try:
        for result in results:
            boxes = result.boxes
            # Find clock in detections
            for _,box in enumerate(boxes):
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
        if latch[0]:
            #We are going to stay in the drop off phase, we will probably have to manually go to shut off or something
            return "DropOff"
        if "person" in class_name:
            #Still have people on board, either travel or path find to them
            return "People"
        if "person" not in class_name:
            #No longer in people finding phase,  now in drop off phase
            latch.append(True)
            latch.pop(0)
            return "DropOff"
    except:
        print("no detections")
     

#Get new coords depending on distance VLM tells us to go
def getCommand(results,xy)->str:
    clock_meta_data = get_clock_box(results)
    try:
        if(clock_meta_data['xywh'][0] < xy['x']):
            return 'D'
        if(clock_meta_data['xywh'][0] > xy['x']):
            return 'B'
        if(clock_meta_data['xywh'][0] > xy['y']):
            return 'L'
        if(clock_meta_data['xywh'][0] > xy['y']):
            return 'R'
        
        return ''
    except:
        return ''


def getCoords(results,response) -> dict: 
    # return [coords["x"], coords["y"]]
    coords = {"x":0,"y": 0}

    clock_meta_data = get_clock_box(results)
    # print(json.loads(response['text']))
    print("TEST:" ,type(response['text']))
    try:
        coords = json.loads(response['text'])
        print("UNPACK",coords)
        coords = {"x": coords['x']+float(clock_meta_data['xywh'][0]),"y":coords['y']+float(clock_meta_data['xywh'][1])}
        return coords
    except:
        print("missed")
        return None

    

def getNewCriticalPoint(results,coords):
    try:
        clock_meta_data = get_clock_box(results)
        print(clock_meta_data)
        if clock_meta_data is not None:
            if((coords[0] > clock_meta_data['xyxy'][0] and coords[0] < clock_meta_data['xyxy'][2]) and (coords[1] > clock_meta_data['xyxy'][1] and coords[1] < clock_meta_data['xyxy'][3])):
                return True
    except:
        return False
        


#TEST BENCH CODE:
# img = cv2.imread(img_path_test)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# model = YOLO('yolov8n.pt')
# results = model(
# source=img, #The camera port
# device=0,  #Cuda device (gpu)
# verbose=False #shuts it up
# )

# getNewCriticalPoint(results,[0,0])

# # for result in  results:
# #     names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
# #     print(names)

# boxes_img = cv2.cvtColor(results[0].plot(),cv2.COLOR_RGB2BGR)
# cv2.imshow("test",boxes_img)
# a=0
# while True:
#     # print("")
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# cv2.destroyAllWindows()
def project_pipeline():
    camera_id = 0
    camera = cv2.VideoCapture(camera_id)
    model = YOLO('yolov8n.pt')
    first_instance=True
    xy = {"x":250,"y":250}
    latch = [False]

    while(True): 
        ret,img = camera.read()
        if not ret:
            print("MISSING CAPTURE: NOT BREAKING -- HOPEFULLY WE RECAPTURE")
        else:

            results = model(
            source=img, #The camera port
            device="cpu",  #Cuda device (gpu)
            verbose=False #shuts it up
            )

            boxed_img =  results[0].plot()
            display(boxed_img)

            if(first_instance or getNewCriticalPoint(results,xy)):
                response = send_to_VLM(img,getPhases(latch,results))
                xy = getCoords(results,response.json())
                if(xy != None):    
                    first_instance = False
            print(xy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            print(latch,getPhases(latch,results))
            # print(xy)
            match(getCommand(results,xy)):
                case 'D':
                    #Drive forward command
                    print("go forward")
                case 'B':
                    #Reverse
                            #   Prompt Robot to ask if we drove to where we are supposed to
                    print("go Backwards")

                case 'R':
                    #turn 90 degrees drive forward
                            #   Prompt Robot to ask if we drove to where we are supposed to
                    print("Turn right go forward")

                case 'L':
                    #turn -90 degrees drive forward
                            #   Prompt Robot to ask if we drove to where we are supposed to
                            #   go  back to original position afterwards.
                    print("turn left go forwards")
                case _:
                    print("BAD DATA")

def test_bench():
    img = Image.open("C:/Users/randy/OneDrive/Desktop/Spatial-VLA/Photos/Lab_photo.jpg")
    img = np.array(img) 

    # 3. Convert RGB to BGR (Crucial for OpenCV compatibility)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    response = send_to_VLM(img,"People")
    print(response)



if __name__ == "__main__":
    # project_pipeline()
    test_bench()




            
    cv2.destroyAllWindows()
