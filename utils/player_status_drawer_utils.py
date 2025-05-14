import cv2 as cv
import numpy as np

def draw_player_status(frames, status):
    
    for idx, row in status.iterrows():
       
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']

        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']


        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']

        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']

        frame = frames[idx]

        shapes = np.zeros_like(frame, np.uint8)
        
        width = 350
        height = 230

        start_x = frame.shape[1] - 400
        start_y = frame.shape[0] - 500

        end_x = start_x + width + 30
        end_y = start_y + height

        overlay = frame.copy()
        cv.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), cv.FILLED)

        alpha = .5

        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        frames[idx] = frame

        txt = 'Player 1     player 2'

        img = cv.putText(
            frames[idx], 
            txt, 
            (start_x + 110, start_y + 30), 
            cv.FONT_HERSHEY_SIMPLEX, 
            .7, 
            (255, 255, 255), 
            2, 
            cv.LINE_AA
        )

        frames[idx] = img

############################################################################################################

        txt = 'Shot Speed'

        img = cv.putText(
            frames[idx], 
            txt, 
            (start_x + 10, end_y - 160),
            cv.FONT_HERSHEY_SIMPLEX, 
            .55, 
            (255, 255, 255), 
            2, 
            cv.LINE_4
        )

        frames[idx] = img

##############################################################################################################

        txt = f' {player_1_shot_speed:.2f} km/h     {player_2_shot_speed:.2f} km/h'

        img = cv.putText(
            frames[idx], 
            txt, 
            (start_x + 110, end_y - 160),
            cv.FONT_HERSHEY_SIMPLEX, 
            .55, 
            (255, 255, 255), 
            2, 
            cv.LINE_AA
        )

        frames[idx] = img

##############################################################################################################

        txt = 'Player Sp.'

        img = cv.putText(
            frames[idx], 
            txt, 
            (start_x + 10, end_y - 120),
            cv.FONT_HERSHEY_SIMPLEX, 
            .55, 
            (255, 255, 255), 
            2, 
            cv.LINE_8
        )

        frames[idx] = img

##############################################################################################################

        txt = f' {player_1_speed:.2f} km/h     {player_2_speed:.2f} km/h'

        img = cv.putText(
            frames[idx], 
            txt, 
            (start_x + 110, end_y - 120),
            cv.FONT_HERSHEY_SIMPLEX, 
            .55, 
            (255, 255, 255), 
            2, 
            cv.LINE_AA
        )

        frames[idx] = img

##############################################################################################################

        txt = 'Avg Sh. Sp.'

        img = cv.putText(
            frames[idx], 
            txt, 
            (start_x + 10, end_y - 80),
            cv.FONT_HERSHEY_SIMPLEX, 
            .55,
            (255, 255, 255), 
            2, 
            cv.LINE_8
        )

        frames[idx] = img

##############################################################################################################

        txt = f' {avg_player_1_shot_speed:.2f} km/h     {avg_player_2_shot_speed:.2f} km/h'

        img = cv.putText(
            frames[idx], 
            txt, 
            (start_x + 110, end_y - 80),
            cv.FONT_HERSHEY_SIMPLEX,            
            .55, 
            (255, 255, 255), 
            2, 
            cv.LINE_AA
        )

        frames[idx] = img

##############################################################################################################

        txt = 'Avg P. Sp.'

        img = cv.putText(
            frames[idx], 
            txt, 
            (start_x + 10, end_y - 40),
            cv.FONT_HERSHEY_SIMPLEX, 
            .55, 
            (255, 255, 255), 
            2, 
            cv.LINE_8
        )

        frames[idx] = img

##############################################################################################################

        txt = f' {avg_player_1_speed:.2f} km/h     {avg_player_2_speed:.2f} km/h'

        img = cv.putText(
            frames[idx], 
            txt, 
            (start_x + 110, end_y - 40),
            cv.FONT_HERSHEY_SIMPLEX, 
            .55, 
            (255, 255, 255), 
            2, 
            cv.LINE_AA
        )

        frames[idx] = img

    return frames