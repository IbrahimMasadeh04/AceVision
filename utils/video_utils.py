import cv2 as cv
def read_video(video_path):

    video = cv.VideoCapture(video_path)

    frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        
        frames.append(frame)

    return frames


def save_video(out_vid, out_path):
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter(out_path, fourcc, 24.0, (out_vid[0].shape[1], out_vid[0].shape[0]))

    for frame in out_vid:
        out.write(frame)
    
    out.release()