# AceVision

AceVision is a computer vision prototype designed to assist in evaluating tennis player movements using pose estimation techniques. It provides real-time feedback on posture and positioning, helping players identify incorrect stances and suggesting better alternatives.

## Project Overview
This prototype was developed as a submission for a computer vision competition. It analyzes the pose and movement of two tennis players on the court to detect:
- Whether players are positioned correctly on their side of the court.
- If their knees are too straight, indicating poor agility.
- Whether their torso posture is too upright, suggesting a lack of readiness.

## key Features
- Uses MediaPipe for pose estimation.
- Identifies and logs improper player poses or alignment.
- Generates text-based feedback with improvement suggestions.
- Supports saving feedback automatically to a chosen folder.

## 🧠 Technologies Used

- **Python**
- **MediaPipe** for pose detection
- **OpenCV** for video frame handling
- **NumPy** for angle calculations
- **Pandas** for ball interpulation
- **Ultralytics** to use YOLO (You Only Look Once)

## Project Structure
AceVision/  
│  
├──constants/  
│    └── `__init__.py`  
│  
├── court_line_detector/  
│    ├── \_\_init\_\_.py  
│    └── court_line_detector.py  
│  
├── input_videos/  
│    └── your_input_videos  
│  
├── mini_court/  
│    ├── \_\_init\_\_.py  
│    └── mini_court.py  
│  
├── models/  
│    ├── best_ball_26_3.pt  
│    ├── new_keypoints_model_4.pth  
│    └── yolov8x.pt  
│  
├── outputs/  
│    └── your_output_videos_where_each_output_is_structured_in_a_separate_folder  
│  
├── pose_est/  
│    ├── \_\_init\_\_.py  
│    └── player_pose_estimadtion.py  
│  
├── store_output/  
│    ├── \_\_init\_\_.py  
│    └── chk_out_path.py  
│  
├── tracker_stubs/  
│    └── preprocessed_detection_result_is_stored_in_a_serialized_format_using_python_`joblib`_module  
│  
├── trackers/  
│    ├── \_\_init\_\_.py  
│    ├── ball_tracker.py  
│    └── player_tracker.py  
│  
├── utils/  
│    ├── \_\_init\_\_.py  
│    ├── bbox_utils.py  
│    ├── conversions.py  
│    ├── player_status_drawer_utils.py  
│    ├── pose_estimation_utils.py  
│    └── video_utils.py  
│  
├── main.py  
│  
├── README.md  
│  
└── requirements.txt  
  
## Detection & Analysis Workflow

- **Player Detection**:  
  The player detection is handled by the pre-trained model `yolov8x.pt`, which is based on [YOLOv8](https://github.com/ultralytics/ultralytics). This model provides real-time detection of players on the court.

- **Ball Detection**:  
  A custom YOLO model is used specifically to detect the tennis ball. The model is designed to track the ball across the frames, ensuring precise localization throughout the game.

- **Court Keypoint Detection**:  
  A custom model, trained using **PyTorch**, detects the key points of the tennis court. This helps in analyzing the players' positioning relative to the court's layout.  
  - **Weights file**: `new_keypoints_model_4.pth`

- **Movement Analysis**:  
  Once the players and the ball are detected, the system analyzes their movements. It uses pose estimation techniques to identify incorrect player positions and posture. This analysis assists in providing feedback for improving player agility and positioning on the court.

- **Preprocessed Detections**:  
  For efficiency, the player and ball detections are serialized using Python's `joblib` module and stored in a `.pkl` file.

## How to Run
1. create a venv
2. activate it
3. Install requirements:
  `pip install -r requirements.txt`
4. Run the main script:
  `python main.py`

## Notes to Take Care about
1. the pre-trained weights are in a separate folder in my drive, here is the link of it: [https://1drv.ms/f/c/c1c38c35788f8718/El103dJyo0ZFoNCJl9J4tVoB_nYtt_KguZ-_Tfw0EkcGgw?e=VCK6rf]
2. load an input video in a new folder named `input_videos`
3. make a folder named `tracker_stubs`, and in `main.py` file, set the parameter `read_from_stub` in `player_detections` and `ball_detections` by true
4. if there is more than one input video, make a folder named `input_video_i`, while `i` means the video number, and then store the `player_detections.pkl` and `ball_detections.pkl` in them
