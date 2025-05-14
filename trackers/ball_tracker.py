from ultralytics import YOLO
import joblib
import cv2 as cv
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_detections):
        ball_positions = [x.get(1, []) for x in ball_detections]
        df_ball_positions = pd.DataFrame(ball_positions)

        # interpolate the missing value
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def get_ball_shots(self, ball_detections):
        ball_positions = [x.get(1, []) for x in ball_detections]

        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions['ball_hit'] = 0
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        min_change_frame_for_hit = 25

        for i in range(len(df_ball_positions) - int(min_change_frame_for_hit * 1.2)):
            negative_pos_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0
            positive_pos_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0

            if negative_pos_change or positive_pos_change:
                change_cnt = 0

                for change_frame in range(i + 1, i + int(min_change_frame_for_hit * 1.2) + 1):
                    negative_pos_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_pos_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_pos_change_following_frame:
                        change_cnt += 1
                    elif positive_pos_change_following_frame:
                        change_cnt += 1

                if change_cnt > min_change_frame_for_hit - 1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hit = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()
        
        return frame_nums_with_ball_hit, df_ball_positions['mid_y_rolling_mean']

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):

        ball_detections = []

        if read_from_stub and stub_path is not None:
            
            ball_detections = joblib.load(stub_path)
            return ball_detections
            
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path is not None:
            joblib.dump(ball_detections, stub_path)

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=.15)[0]
        
        ball_dict = {}

        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result # ball is always at index 1

        return ball_dict
    
    def draw_bboxes(sefl, vid_frames, ball_detections):
        out_vid_frames = []

        for frame, ball_dict in zip(vid_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv.rectangle(
                    frame, 
                    (int(x1), int(y1)), 
                    (int(x2), int(y2)), 
                    (255, 0, 0), 
                    2
                )

                cv.putText(
                    frame, 
                    str(f"Ball ID: {track_id}"), 
                    (int(x1), int(y1 - 10)), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (255, 0, 0), 
                    2    
                )

            out_vid_frames.append(frame)

        return out_vid_frames