import argparse
from inference import InferencePipeline
import cv2 as cv
import numpy as np
from inference.core.interfaces.stream.sinks import render_boxes
from inference import get_model
import supervision as sv
from inference.core.utils.image_utils import load_image
from PIL import ImageGrab
parser = argparse.ArgumentParser(description="YoloV8 Inference with camera/zoom implementation")
parser.add_argument("-v","--video",type=str, nargs=1, default="/")
parser.add_argument("--path",action='store_true')
parser.add_argument("-p","--port", nargs="?", default=0)
args = parser.parse_args()
# print(args.video[0])

if args.video:
    if(args.path is None ):
        path = "../Videos/"+args.video[0]
    else:
        path = args.video[0]
    print(path)
    capture = cv.VideoCapture(path)
    ret,frame= capture.read()
    model = get_model(model_id="yolov8n-640")
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        image = frame
        results = model.infer(image)[0]
        results = sv.Detections.from_inference(results)
        annotator = sv.BoxAnnotator(thickness=2)
        annotated_image = annotator.annotate(image,results)
        annotator = sv.LabelAnnotator(text_scale=1,text_thickness=1)
        annotated_image = annotator.annotate(image,results)
        cv.imshow("Video",annotated_image)
        if(cv.waitKey(1) & 0xFF == ord('q')):
            capture.release()
            cv.destroyAllWindows()
else:
    print("Using Camera port: ["+str(args.port)+"]")
    pipeline = InferencePipeline.init(
    model_id="yolov8n-640",
    video_reference=args.port,
    on_prediction=render_boxes
    )
    pipeline.start()
    pipeline.join()
