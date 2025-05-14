from utils import read_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2 as cv
from store_output import save

from utils import measure_distance, convert_pixel_distance_to_meters, draw_player_status
import constants
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from pose_est import PlayerPoseEstimator

def main():

    # video path
    video_path = 'input_videos/input_video_2.mp4'

    # read video frames
    video_frames = read_video(video_path)

    # detect players and ball
    player_tracker = PlayerTracker('models/yolov8x.pt')
    ball_tracker = BallTracker('models/best_ball_26_3.pt')

    player_detections = player_tracker.detect_frames(
        video_frames, 
        read_from_stub=True, 
        stub_path="tracker_stubs/input_video_2/player_detections.pkl"
    )

    ball_detections = ball_tracker.detect_frames(
        video_frames, 
        read_from_stub=True,
        stub_path="tracker_stubs/input_video_2/ball_detections.pkl"
    )

    # interpolate ball positions
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # court line detector model
    court_model_path = 'models/new_keypoints_model_4.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players
    player_detections = player_tracker.filter_players_by_id(player_detections)

    # Mini court
    mini_court = MiniCourt(video_frames[0])

    # detect ball shots
    ball_shots_frames, rolling_mean = ball_tracker.get_ball_shots(ball_detections)
    print(ball_shots_frames)

    # convert positions to mini court positions
    player_mini_court_detections = mini_court.convert_bboxes_to_mini_court_coordinates_player(player_detections, court_keypoints)
    ball_mini_court_detections = mini_court.convert_bboxes_to_mini_court_coordinates_ball(ball_detections, court_keypoints)
                                                                                          
    # set the player and ball status data
    player_status_data = [{
        'frame_number': 0,

        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,
        
        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0
    }]

    for ball_shot_idx in range(len(ball_shots_frames) - 1):
        
        start_frame = ball_shots_frames[ball_shot_idx]
        end_frame = ball_shots_frames[ball_shot_idx + 1]

        ball_shot_time_in_seconds = (end_frame - start_frame) / 24 # 24 fps

        # the distance covered by the ball from start frame to end frame
        ball_distance_in_px = measure_distance(ball_mini_court_detections[start_frame][1], 
                                          ball_mini_court_detections[end_frame][1])
        
        ball_distance_in_meters = convert_pixel_distance_to_meters(ball_distance_in_px, 
                                                                   constants.DOUBLE_LINE_WIDTH, 
                                                                   mini_court.get_width_of_mini_court())
        
        # the speed of the ball in km/h
        ball_speed_shots = ball_distance_in_meters / ball_shot_time_in_seconds * 3.6

        # the player who shot the ball
        player_pos = player_mini_court_detections[start_frame]
        player_shot_ball = min(player_pos.keys(), key=lambda player_id: measure_distance(player_pos[player_id], 
                                                                                         ball_mini_court_detections[start_frame][1]))
        
        # opponent player id
        opponent_player = 2 if player_shot_ball == 1 else 1
        
        # opponent player position
        distance_covered_by_opponent_in_px = measure_distance(player_mini_court_detections[start_frame][opponent_player],
                                                        player_mini_court_detections[end_frame][opponent_player])
        
        distance_covered_by_opponent_in_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_in_px, 
                                                                constants.DOUBLE_LINE_WIDTH, 
                                                                mini_court.get_width_of_mini_court())
        

        opponent_player_speed = distance_covered_by_opponent_in_meters / ball_shot_time_in_seconds * 3.6

        curr_player_status = deepcopy(player_status_data[-1]) # copy the last player status

        curr_player_status['frame_number'] = start_frame
        curr_player_status[f'player_{player_shot_ball}_number_of_shots'] += 1
        curr_player_status[f'player_{player_shot_ball}_total_shot_speed'] += ball_speed_shots
        curr_player_status[f'player_{player_shot_ball}_last_shot_speed'] = ball_speed_shots

        curr_player_status[f'player_{opponent_player}_total_player_speed'] += opponent_player_speed
        curr_player_status[f'player_{opponent_player}_last_player_speed'] = opponent_player_speed

        player_status_data.append(curr_player_status)

    
    player_status_data_df = pd.DataFrame(player_status_data)
    
    frames_df = pd.DataFrame({'frame_number': list(range(len(video_frames)))})

    player_status_data_df = pd.merge(frames_df, player_status_data_df, how='left', on='frame_number')
    player_status_data_df = player_status_data_df.ffill()

    player_status_data_df['player_1_average_shot_speed'] = player_status_data_df['player_1_total_shot_speed'] / player_status_data_df['player_1_number_of_shots']
    player_status_data_df['player_2_average_shot_speed'] = player_status_data_df['player_2_total_shot_speed'] / player_status_data_df['player_2_number_of_shots']

    player_status_data_df['player_1_average_player_speed'] = player_status_data_df['player_1_total_player_speed'] / player_status_data_df['player_1_number_of_shots']
    player_status_data_df['player_2_average_player_speed'] = player_status_data_df['player_2_total_player_speed'] / player_status_data_df['player_2_number_of_shots']

    # estimate the players poses
    player_pose_estimator = PlayerPoseEstimator()

    out_video_frames, player_1_poses = player_pose_estimator.detect_player_pose(video_frames, player_detections, player_key=1)
    out_video_frames, player_2_poses = player_pose_estimator.detect_player_pose(out_video_frames, player_detections, player_key=2)

    # draw bboxes
    out_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    out_video_frames = ball_tracker.draw_bboxes(out_video_frames, ball_detections)
    
    # draw court keypoints
    out_video_frames = court_line_detector.draw_keypoints_on_video(out_video_frames, court_keypoints)

    # draw mini court
    out_video_frames = mini_court.draw_mini_court(out_video_frames)
    out_video_frames = mini_court.draw_points_on_mini_court(out_video_frames, player_mini_court_detections)
    out_video_frames = mini_court.draw_points_on_mini_court(out_video_frames, ball_mini_court_detections, color=(255, 0, 0))

    # draw player status
    out_video_frames = draw_player_status(out_video_frames, player_status_data_df)

    # draw frame number on the top-left corner
    for i, frame in enumerate(out_video_frames):
        
        cv.rectangle(
            frame,
            (10, 100),
            (220, 140),
            (255, 255, 255),
            cv.FILLED
        )

        cv.putText(
            frame, 
            f'Frame {i}', 
            (30, 130), 
            cv.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), 
            2, 
            cv.LINE_AA
        )
    
    # save video

    folder_path = save(out_video_frames, 'outputs')

    # detect the wrong moves
    player_pose_estimator.dertect_wrong_moves(
        player_1_poses, 
        player_2_poses, 
        player_detections, 
        court_keypoints, 
        folder_path
    )

    # save ball shots as an img

    plt.figure()
    rolling_mean.plot()
    plt.savefig(f"{folder_path}/ball_status.png", bbox_inches='tight', dpi=300)
    
    player_pose_estimator.close()
    plt.close()

if __name__ == "__main__":
    main()