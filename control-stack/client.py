
#!/usr/bin/env python3
"""
Quick test - sends a small red square image to your VLM server
"""

import requests
import base64
import io
import cv2
from PIL import Image

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
            "prompt": "How many men are there",
            "max_tokens": 50,
            "temperature": 0.7
        }
        response = requests.post(
            f"{SERVER_URL}/inference",
            json=payload,
            timeout=30
        )
        print(response.json())
        