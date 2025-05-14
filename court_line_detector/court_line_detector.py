import torch
from torchvision import models
from torchvision.transforms import transforms
import cv2 as cv

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2) # 14 points X 2 coordinates
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def predict(self, frames):
        

        img_rgb = cv.cvtColor(frames, cv.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(img_tensor)

        kps = outputs.squeeze().cpu().numpy()

        original_h, original_w = frames.shape[:2]
        
        kps[::2] *= original_w / 224.0
        kps[1::2] *= original_h / 224.0

        return kps
    
    def draw_keypoints(self, image, keypoints):
        
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv.putText(image, str(i // 2), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.circle(image, (x, y), 5, (0, 0, 255), -1)

        return image
    
    def draw_keypoints_on_video(self, vid_frames, kps_list):
        
        out_vid_frames = []

        for frame in vid_frames:
            frame = self.draw_keypoints(frame, kps_list)
            out_vid_frames.append(frame)

        return out_vid_frames