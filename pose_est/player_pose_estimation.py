import mediapipe as mp
import cv2 as cv
from utils import calculate_angle, get_foot_position, get_closest_kp_to_player
from store_output import log_pose_feedback

class PlayerPoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

        self.logs_info = []
    
    def detect_player_pose(self, frames, player_detections, player_key):

        output_frames = []
        landmarks_list = []

        for i, frame in enumerate(frames):
            
            
            # Get the bounding box of the palyers
            player_bbox = player_detections[i][player_key]
            x1, y1, x2, y2 = int(player_bbox[0]), int(player_bbox[1]), int(player_bbox[2]), int(player_bbox[3])


            # Crop the frame to the closest player's bounding box
            player_crop = frame[y1:y2, x1:x2]

            PAD = 20
            player_crop = frame[max(y1 - PAD, 0):min(y2 + PAD, frame.shape[0]),
                                max(x1 - PAD, 0):min(x2 + PAD, frame.shape[1])]

            # Process the pose of the cropped player
            image_rgb = cv.cvtColor(player_crop, cv.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = [
                    {
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility
                    }
                    for lm in results.pose_landmarks.landmark
                ]
                landmarks_list.append(landmarks)

                # Draw pose landmarks on the frame (for visualization)
                self.draw_pose_full_frame(frame, results, player_bbox)

            else:
                print('No landmarks detected for player 1.')


            # Store the processed frame
            output_frames.append(frame)

        return output_frames, landmarks_list

    def draw_pose_full_frame(self, frame, results, actual_bbox):
        x1, y1, x2, y2 = actual_bbox

        cropped_width = x2 - x1
        cropped_height = y2 - y1

        if not results.pose_landmarks:
            raise ValueError('No landmarks detected')

        # draw connections between landmarks
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection

            start = results.pose_landmarks.landmark[start_idx]
            end = results.pose_landmarks.landmark[end_idx]

            start_x = int(start.x * cropped_width + x1)
            start_y = int(start.y * cropped_height + y1)
            end_x = int(end.x * cropped_width + x1)
            end_y = int(end.y * cropped_height + y1)

            cv.line(
                frame, 
                (start_x, start_y), 
                (end_x, end_y), 
                (0, 255, 255), 
                2
            )

        # draw joints (landmarks)

        for lm in results.pose_landmarks.landmark:
            cx = int(lm.x * cropped_width + x1)
            cy = int(lm.y * cropped_height + y1)
            
            cv.circle(
                frame, 
                (cx, cy), 
                5, 
                (0, 0, 255), 
                -1
            )

    def add_log(self, issue, recommendation):
        log_entry = {issue: recommendation}
        if log_entry not in self.logs_info:
            self.logs_info.append(log_entry)

    def dertect_wrong_moves(self, player_1_poses, player_2_poses, player_detections, court_keypoints, saving_folder_path):

        # Analyze the pose landmarks to detect wrong moves

        for i, (player_1_pose, player_2_pose) in enumerate(zip(player_1_poses, player_2_poses)):
            
            player_1_bbox = player_detections[i][1]
            player_2_bbox = player_detections[i][2]
            
            # Get the player's foot position
            player_1_foot_position = get_foot_position(player_1_bbox)
            player_2_foot_position = get_foot_position(player_2_bbox)
            
            # know the closest keypoint index to the foot position
            
            player_1_kp_idx = get_closest_kp_to_player(player_1_foot_position, court_keypoints, [2, 5, 7, 3])
            player_2_kp_idx = get_closest_kp_to_player(player_2_foot_position, court_keypoints, [0, 4, 6, 1])

            # if the players in the same court side, add to a text file the wrong move, and the recommended move to take care of

            if ((player_1_kp_idx in [2, 5]) and (player_2_kp_idx in [0, 4])):
                
                self.add_log(
                    'The Opponents Are Facing Each Other In One Line (Left Side)',
                    'Player 1 or 2 should move to the right side of the court'
                )
                
            elif ((player_1_kp_idx in [3, 7]) and (player_2_kp_idx in [1, 6])):
                
                self.add_log(
                    'The Opponents Are Facing Each Other In One Line (Right Side)',
                    'Player 1 or 2 should move to the left side of the court'
                )
                

############################################################################################################

            # chk if the leg is too straight or not
            '''player 1 legs'''
            player_1_right_hip = player_1_pose[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            player_1_right_knee = player_1_pose[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
            player_1_right_ankle = player_1_pose[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]

            player_1_left_hip = player_1_pose[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            player_1_left_knee = player_1_pose[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
            player_1_left_ankle = player_1_pose[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]

            '''player 2 legs'''
            player_2_right_hip = player_2_pose[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            player_2_right_knee = player_2_pose[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
            player_2_right_ankle = player_2_pose[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]

            player_2_left_hip = player_2_pose[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            player_2_left_knee = player_2_pose[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
            player_2_left_ankle = player_2_pose[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]

            # Calculate the angles for player 1
            player_1_right_angle = calculate_angle(player_1_right_hip, player_1_right_knee, player_1_right_ankle)
            player_1_left_angle = calculate_angle(player_1_left_hip, player_1_left_knee, player_1_left_ankle)

            # Calculate the angles for player 2
            player_2_right_angle = calculate_angle(player_2_right_hip, player_2_right_knee, player_2_right_ankle)
            player_2_left_angle = calculate_angle(player_2_left_hip, player_2_left_knee, player_2_left_ankle)


            if player_1_left_angle > 170 and player_1_right_angle > 170:
                
                self.add_log(
                    
                    'Player 1 is too straight',
                    'Bend the knees for better agility'
                    
                )

            elif player_1_left_angle > 170 or player_1_right_angle > 170:

                self.add_log(
                    'Player 1: One knee is too straight',
                    'try to bend both knees equally for better balance'    
                )
                

            if player_2_left_angle > 170 and player_2_right_angle > 170:
                
                self.add_log(    
                    'Player 2 is too straight',
                    'Bend the knees for better agility'
                )
                

            elif player_2_left_angle > 170 or player_2_right_angle > 170:

                self.add_log(    
                    'Player 2: One knee is too straight',
                    'try to bend both knees equally for better balance'
                )
                

#############################################################################################################

            # chk if is leaning forward slightly
            player_1_left_shoulder = player_1_pose[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            player_1_right_shoulder = player_1_pose[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
            # player 1 hip and knee for left and right sides are already exist

            player_2_left_shoulder = player_2_pose[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            player_2_right_shoulder = player_2_pose[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
            # player 2 hip and knee for left and right sides are already exist
            
            # Calculate the torso angles for both players
            # begin with the left side for both players
            player_1_left_torso_angle = calculate_angle(player_1_left_shoulder, player_1_left_hip, player_1_left_knee)
            player_2_left_torso_angle = calculate_angle(player_2_left_shoulder, player_2_left_hip, player_2_left_knee)
            
            # now the right side for both players
            player_1_right_torso_angle = calculate_angle(player_1_right_shoulder, player_1_right_hip, player_1_right_knee)
            player_2_right_torso_angle = calculate_angle(player_2_right_shoulder, player_2_right_hip, player_2_right_knee)

            if player_1_left_torso_angle > 150 and player_1_right_torso_angle > 150:

                self.add_log(    
                    'Player 1 is too straight',
                    'Lean your upper body slightly forward to maintain readiness'
                )
                

            elif player_1_left_torso_angle > 150 or player_1_right_torso_angle > 150:
                
                self.add_log(
                    'Player 1 is leaning too much to one side',
                    'Lean your upper body slightly forward to maintain readiness'
                )
                
            if player_2_left_torso_angle > 150 and player_2_right_torso_angle > 150:
                
                self.add_log(
                    'Player 2 is too straight', 
                    'Lean your upper body slightly forward to maintain readiness'
                )

            elif player_2_left_torso_angle > 150 or player_2_right_torso_angle > 150:

                self.add_log(
                    'Player 2 is leaning too much to one side', 
                    'Lean your upper body slightly forward to maintain readiness'
                )
        
        # saving the feedback to a text file
        for info in self.logs_info:
            for issue, recommendation in info.items():
                log_pose_feedback(saving_folder_path, issue, recommendation)

    def close(self):
        self.pose.close() 