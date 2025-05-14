import os
from utils import save_video

def get_name(folder, base_name='output', ext='.avi', file=False):
    i = 1
    if file:
        while os.path.exists(os.path.join(folder, f"{base_name}_{i}{ext}")):
            i += 1
    else:
        while os.path.exists(os.path.join(folder, f"{base_name}_{i}")):
            i += 1

    return os.path.join(folder, f"{base_name}_{i}{ext}") if file else os.path.join(folder, f"{base_name}_{i}")

def save(frames, folder, base_name='output', ext='.avi'):
    os.makedirs(folder, exist_ok=True)
    new_folder = get_name(folder)
    
    os.makedirs(new_folder)
    
    file_path = get_name(new_folder, base_name, ext, file=True)
    save_video(frames, file_path)
    print(f"Video saved in: {new_folder}")
    return new_folder

def log_pose_feedback(folder_name, issue, recommendation, log_file="pose_feedback_log.txt"):
    
    # chk if the file exists
    # if not, create it and write the header

    path = f"{folder_name}/{log_file}"
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write("=== Player Pose Feedback Log ===\n\n")

    # append the feedback to the log file
    with open(path, 'a') as f:
        
        f.write(f'❌ Issue: {issue}.\n')
        f.write(f'💡 Tip: {recommendation}.\n\n')