#YoloV12 Environment & Code
##Description:
###This foler contains the environment to run yolov12 model: "yolov12env"
###It also contains a code file to extract frames from videos, then inference the position of objects in each frame.
###The frames and the data will be stored automatically in "output/frames;output/lables".
###The code can also lable single image.
##How to run?
###First source the yolo environment: source YoloV12/yolov12env/bin/activate
###To inference image/video, run: YoloV12/detect_and_extract "path of your source"