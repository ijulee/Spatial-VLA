
#!/usr/bin/env python3
"""
Quick test - sends a small red square image to your VLM server
"""

import requests
import base64
import io
import cv2
from PIL import Image
import time


SERVER_URL = "https://ik92uwhwu2vm2v-8000.proxy.runpod.net"

img_path_test = "C:/Users/randy/Desktop/Spatial-VLA/Photos/scene_001_bus_06.png"
import cv2 
from ultralytics import YOLO


def send_to_VLM(img) :
    # img = cv2.imread(img_path_test,cv2.IMREAD_COLOR)
    if img is not None:
        sucess,buffer = cv2.imencode('.jpg',img)

        if sucess:
            img_bytes = buffer.tobytes()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            # try:

            payload = {
                "image": img_b64,
                "prompt": "in this format and only this format (NO TEXT AT ALL) {'X' : x_val, 'Y' : y_val} give me the coordinates of the bench closest to the man on the top right",
                "max_tokens": 1000000, #MESS AROUND WITH THIS
                "temperature": 1.0
            }
            # print(payload["image"])
            start = time.perf_counter()
            response = requests.post(
                f"{SERVER_URL}/inference",
                json=payload
                # timeout=30
            )
            timing = time.perf_counter() - start
            print(response.json())
            print("RESPONSE TIME: " , timing, " seconds" )
        return response.json()
    return None

# def MetaData(results):
    
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             print(box.cid)

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
                    'center': (float(xywh[0]), float(xywh[1])),
                    'confidence': confidence,
                    'bbox': [float(x) for x in xyxy]  # For collision detection
                }
    
    return None

def getCoords(response):
    coords = response["text"]
    return [coords["x"], coords["y"]]

def getNewCriticalPoint(results,coords):
    clock_meta_data = get_clock_box(results)
    print(clock_meta_data)
    if(coords[0] < clock_meta_data['xyxy'][0] or coords[0] > clock_meta_data['xyxy'][2] or coords[1] > clock_meta_data['xyxy'][1] or coords[1] > clock_meta_data['xyxy'][3]):
        return True
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
    
camera_id = 0
camera = cv2.VideoCapture(camera_id)
model = YOLO('yolov8n.pt')
first_instance=True
while(True): 
    ret,img = camera.read()
    if not ret:
        print("MISSING CAPTURE: NOT BREAKING -- HOPEFULLY WE RECAPTURE")
    else:
        results = model(
        source=img, #The camera port
        device=0,  #Cuda device (gpu)
        verbose=False #shuts it up
        )

        boxed_img =  cv2.cvtColor(results[0].plot(),cv2.COLOR_RGB2BGR)

        cv2.imshow("CAMERA VIEW", boxed_img) #Just display for live demo purposes i guess
        
        if(first_instance or getNewCriticalPoint(xy,boxed_img)):
            response = send_to_VLM(img)
            xy = getCoords(response)
            first_instance = False
        




    












    