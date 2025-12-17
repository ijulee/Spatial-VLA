# from inference import InferencePipeline
# from inference.core.interfaces.stream.sinks import render_boxes
#  #plug and play v8 inference
# pipeline = InferencePipeline.init(
#     model_id="yolov8n-640",
#     video_reference=0,
#     on_prediction=render_boxes
# ),
# pipeline.start()
# pipeline.join()


#To do: 
# Move inference to the gpu (my cpu is maxing out)
# Find collisions
# Snapshot collisions
# Weight the collisions and send to VLM
# draw next critcal point -- or just save in memory lol

#we will probably move to ultralytics, but now we are expecting lag lol
import cv2 as cv
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
camera = cv.VideoCapture(0)

def MetaData(results):
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            

class_names = model.names
for class_id, class_name in class_names.items():
    print(f"Class ID {class_id}: {class_name}")
print("=" * 50 + "\n")
while True:
    err,img = camera.read()
    if not err:
        print("ERR")
        break

    results = model(
        source=img, #The camera port
        device=0,  #Cuda device (gpu)
        verbose=False #shuts it up
    )
    

    render_boxes = results[0].plot()
    
    cv.imshow("frames",render_boxes)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

