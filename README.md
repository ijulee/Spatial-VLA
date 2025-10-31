Project Title: Spatial Reasoning enhanced VLAs
Group Members: I-Jui Lee, Randy Bui, Ben Davis, Ivan Xu
Goals
This project will seek to train an improved Vision-Language-Action (VLA) model with spatial reasoning capabilities for a mobile robot in a controlled environment. Given that most of the state-of-art VLAs lack a specific training stage for spatial reasoning capabilities, this project tries to fulfill this gap.  Specifically, we plan to create at least 10 different visual question-answer (VQA) templates which embed certain rules and relationships within the environment. These templates will include action-based rules defined by us for things like navigation (e.g., where should the robot move based on visual observations?), as well as quantitative rules, such as determining the distances between the robot and other visible objects. These templates, as well as a collection of images collected in our environment, will be used to generate a dataset of image-question-answer pairs using GRAID. Finally, this dataset will be used to fine-tune a vision-language model (VLM) with techniques like supervised fine-tuning (SFT) or reinforcement learning (RL), giving it some sense of spatial reasoning in our environment. 
Approach
Step 1: Training Environment Setup
Robot moving site: A 2m*2m square site
Robot car: 
Calibrate movement within our set environment using IMU feedback
Set up wireless connectivity through which we can relay movement instructions
Obstacles： colored balls, colored cubes, stickers, denoters
Rules: manually defined, such as:
IF nearest(red_ball) & distance<0.5m THEN turn_left(30°)
IF see(green_block) THEN turn_right(30°)
IF see(mushroom_sticker) THEN set_speed = 1.5×
Step 2: Camera Calibration
Fix a bird-eye view camera at one of the corners of the square site that robot moves on 
Collect random sets of coordinates pairs in the form of:        [(world_coord_x, world_coord_y), (pixel_coord_x, pixel_coord_y)]
Use these data to calculate a homography projection matrix that helps convert pixel coord. to world coord. or in reverse direction
Step 3: Data Collection:
Design multiple sets of robot moving site layouts (probably 10-20 sets)	
For each set of layout, shoot a corresponding video that the robot randomly moves within that layout.
Use python code to extract images out of those videos
Label the images with descriptions of the following format, stored in a .txt file:
<class_id> <x_center> <y_center> <width> <height>
Item
bbox-coordinates
robot
[0.512, 0.700, 0.120, 0.180]
red_ball
[0.250, 0.450, 0.080, 0.090]
green_block
[0.800, 0.520, 0.100, 0.120]
mushroom_sticker
[0.250, 0.450, 0.080, 0.090]

We could organize the dataset to follow the structure below:
			dataset/
  |── train/
  |	|── images/
  |	|	|── img_001.jpg
  |	|	|── img_002.jpg
  |	|── labels/
  |	|	|── img_001.txt
  |	|	|── img_002.txt
              |── test/
	  |         |...same as train
	  |──classes.txt (label explanation)
Step 4: Train models for object detection
Set up a YOLO8 training script.
Step 5: Creating Spatial Reasoning QA Dataset Using GRAID
Create Question template:
Typical	Questions:
Reasoning Types
Typical Questions
Answer Types
Positional Relationships
Is the red ball on the left/right/front/back of the robot?
Yes/No
Distance Comparison
Which object is closest to the robot?
Object name
Quantitative Distance
How far is the robot from the red ball(in meters)?
Numeral value
Quantitative Reasoning
How many obstacles are in front of the robot?
Quantity
Angular reasoning
What is the direction angle of the red ball relative to the robot? 
Numerical values
Existence 
Is there any green block in the scene?
Yes/No
Relative Position
Is the blue cube between the robot and the red ball?
Yes/No



Collect more robot moving site layouts
Use well-trained Yolov8 model to do object position identification
Use calibrated homography projection matrix to convert pixel position to world position
Write a python script to calculate relative angle/distance between objects
Write a python script to iteratively analyze object pairs and generate Q/A pairs following the above templates.
Finally store the pairs of [Image, Q/As] as a GRAID dataset. We could store it using JSON format like:
[
  {
    "image": "scene_03_frame_001.jpg",
    "qa": [
      {"q": "Is the red_ball on the left side of the robot?", "a": "No"},
      {"q": "How far is the red_ball from the robot (in meters)?", "a": 0.42},
      {"q": "What is the direction angle of the red_ball relative to the robot?", "a": 18.6}
    ]
  },
  ...
]
Step 7: Fine tune VLM Using GRAID Dataset
Potential basic VLMs:LLaVA 1.6 / 1.7, OpenFlamingo / IDEFICS, OpenVLA, π0 / π0.5.
Download basic VLM frameworks and VLM itself
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
conda create -n llava python=3.10
conda activate llava
pip install -e .
huggingface-cli download liuhaotian/LLaVA-1.6-Vicuna-7B

Prepare dataset 
		dataset/
  |── images/
  |	|──scene_03_frame_001.jpg
  |	|	……
  |	|	……
  |── data/
  |	|──spatial_reasoning_qa.json/
  |── finetune.sh
Run the finetune bash (from the official websites)
Evaluation and test
Step 8: Extend VLM to a VLA
Finish this if we have enough time  after the above procedure.
Step 9: Benchmarking:
We will be aiming for both accuracy and time from query to answer.
We would want a speed efficiency matching or beating LIBERO. 
	
Topics from lecture
control/feedback for the robot-server interaction
UART/ peripheral to computer, MMIO, Baud Rate
Calibration (affine functions)
(Possibly) Parallelism, if we want to use the GPU to process our realtime feed faster
(possibly) if we do use parallelism we more than likely will need some scheduling/locking to prevent race conditions between VLA and any processing we do
Finite state machines (computer calling api getting data, using that data to determine the state of the robot)
Resources
GRAID
Camera – Victure SC30 Webcam v1.0(?)
Camera tripod for birds-eye style view
An existing VLA model such as OpenVLA or π0 to be modified
A robot car to navigate our created environment, such as the Pololu 3pi+ 2040
A Wi-Fi or Bluetooth module to add wireless connectivity to relay instructions to the robot car. Example: ESP8266 WiFi module
Colored blocks and sticker sets to use as objects for the VLA to recognize
Schedule
October 24: Project proposal (this document)
October 31: Updated proposal after discussion with GSIs.
November 7: BoM/Materials received after finalized proposal
November 14:  Creating the tabletop benchmark including peripherals; implement object detection/classification
November 21: Creating our database
November 28: SFL and RL on the VLA
December 5:  Setup Robot-Computer system
December 12: Benchmarking.
December 19: Demonstration video made, powerpoint prepared.
December 20: Final presentation and demo.
December 22: Project report and video turned in.
Risks
This project has high computational demand, both for model fine-tuning, as well as for real-time inference. During inference, the process of capturing an image, transmitting it to a computer for computation, and returning an action to the robot must be quick enough to prevent collisions and avoid acting on stale data. This means we must also ensure that our real-time communication speeds are sufficient. Depending on the complexity of our QA pairs and the variability of the scene, this project may also require significant effort during data collection to form vision-question-action pairs. 
Link to GitHub/GitLab repository
https://github.com/ijulee/Spatial-VLA
