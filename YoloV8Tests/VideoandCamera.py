import argparse
import queue
import threading
from av import VideoFrame
from inference import InferencePipeline
import cv2 
from inference.core.interfaces.stream.sinks import render_boxes
import supervision as sv
from typing import Any, List
from ultralytics import YOLO
from PIL import ImageGrab
import mss
class LiveSink:
    def __init__(self, window_name="Live Inference"):
        self.window_name = window_name
        self.queue = queue.Queue(maxsize=2)
        self.running = True
        self.thread = threading.Thread(target=self._display_thread, daemon=True)
        self.thread.start()

    def write_frame(self, frame):
        # non-blocking push (drop old frames to stay low-latency)
        if not self.queue.full():
            self.queue.put(frame)

    def _display_thread(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        while self.running:
            try:
                frame = self.queue.get(timeout=0.1)
                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
            except queue.Empty:
                continue


    def stop(self):
        self.running = False
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()



class wrapper:


    def __init__(self, weights_path: str):
        self._model = YOLO(weights_path)


    # after v0.9.18  
    def infer(self, video_frames: List[VideoFrame]) -> List[Any]: 
    # result must be returned as list of elements representing model prediction for single frame
    # with order unchanged.
        return self._model([v.image for v in video_frames])






box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
def my_render(prediction: dict, video_frame: VideoFrame) -> None:
        # prediction = prediction.to_json()
        results = sv.Detections.from_ultralytics(prediction)
        frame = video_frame.image 
        labels = [
                f"{model._model.model.names[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(results.class_id, results.confidence)
        ]
        annotated = box_annotator.annotate( # type: ignore
            scene=frame,
            detections=results
        )
        annotated = label_annotator.annotate( # type: ignore
            scene=annotated,
            detections=results, 
            labels=labels
        )
        
        sink.write_frame(annotated)

    


parser = argparse.ArgumentParser(description="YoloV8 Inference with camera/zoom implementation")
parser.add_argument("-v","--video",type=str, nargs=1)
parser.add_argument("--path",action='store_true')
parser.add_argument("--custom",action='store_true')
parser.add_argument( "--model",type=str, nargs=1, default="yolo11n.pt")
parser.add_argument("-p","--port", type=str, nargs="?", default=0)
parser.add_argument("-m", "--monitor", type=int, nargs="?")
parser.add_argument("--train", nargs=1 )
args = parser.parse_args()

if args.train is not None:
    model = YOLO(args.model[0]) 
# Train the model on the COCO8 dataset for 100 epochs
    model.train(
    data=args.train,  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )

if args.video is not None:
    if args.path:
        path = args.video
    else:
        path = "../Videos/"+str(args.video[0])
    # model = get_model(args.model[0])
    print("Inferring on: "  + path + " using " + ( str(args.model[0]) if args.custom  else  "yolov8n-640" )   )
    if(args.custom):
        model = wrapper(args.model[0])
        sink = LiveSink()
        pipeline = InferencePipeline.init_with_custom_logic(
            video_reference=path,
            on_video_frame=model.infer,
            on_prediction=my_render
        )

        with sink:
            pipeline.start()
            pipeline.join()
            sink.stop()


else:
    if args.monitor is not None: 
            sct = mss.mss()
            print("hwello")

            # while(True):
            #     img = sct.grab(sct.monitors[1])
            #     cv2.imshow(img)
            #     if(cv2.waitKey(1) & 0xFF == ord("q")):
            #         break
            pipeline = InferencePipeline.init(
                video_reference=sct.grab(sct.monitors[1]),
                model_id="yolov8n-640",
                on_prediction=render_boxes
            )
    else:
        pipeline = InferencePipeline.init(
                video_reference=args.port,
                model_id="yolov8n-640",
                on_prediction=render_boxes
            )
        
    
        pipeline.start()
        pipeline.join()

cv2.destroyAllWindows()
