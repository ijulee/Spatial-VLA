import queue
import threading
from av import VideoFrame
import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv
from inference import InferencePipeline, get_model
from inference.core.interfaces.stream.sinks import  render_boxes

# from supervision.Detections.from_inference import 
from typing import List,Any

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



class MyModel:

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
            f"{my_model._model.model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(results.class_id, results.confidence)
        ]
    annotated = box_annotator.annotate(
       scene=frame,
       detections=results
    )
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=results, 
        labels=labels)
    
    sink.write_frame(annotated)

    


video_reference = "../Videos/People.mp4"
# video_info = sv.VideoInfo.from_video_path(video_reference)

sink = LiveSink()

my_model = MyModel("yolov8s.pt")

pipeline = InferencePipeline.init_with_custom_logic(
  video_reference="../Videos/People.mp4",
  on_video_frame=my_model.infer,
  on_prediction=my_render,
)

# # model = get_model("yolov8n-640")
# pipeline = InferencePipeline.init(
#    model_id="yolov8n-640",
#    video_reference="../Videos/People.mp4",
#    on_prediction=render_boxes
# )
# sv.plot_image(results[0].plot())
# json_string = results.to_json()
# json_string = json.loads(json_string)
# for r in json_string:
#     print(r["name"])
# sv.Detections.from_inference(results['orig_img'])
with sink:
    pipeline.start()

    pipeline.join()