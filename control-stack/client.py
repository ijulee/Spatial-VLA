
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

img_path_test = "C:/Users/randy/Desktop/Spatial-VLA/Photos/Lab_photo.jpg"

img = cv2.imread(img_path_test,cv2.IMREAD_COLOR)

if img is not None:
    sucess,buffer = cv2.imencode('.jpg',img)

    if sucess:
        img_bytes = buffer.tobytes()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        # try:

        payload = {
            "image": img_b64,
            "prompt": "in this format and only this format (NO TEXT AT ALL) {'X' : x_val, 'Y' : y_val} give me the coordinates of the bench closest to the man on the top right",
            "max_tokens": 100,
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