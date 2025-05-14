def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return center_x, center_y

def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def get_foot_position(bbox):
    return (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))

def get_closest_keypoint_index(point, kps, kps_idxs):
    closest_dist = float('inf')
    kp_idx = kps_idxs[0]

    for idx in kps_idxs:
        kp = (kps[idx * 2], kps[idx * 2 + 1])
        dist = abs(point[1] - kp[1])
        
        if dist < closest_dist:
            closest_dist = dist
            kp_idx = idx

    return kp_idx
def measure_y_distance(p1, p2):
    return abs(p1[1] - p2[1])

def measure_x_distance(p1, p2):
    return abs(p1[0] - p2[0])

def get_height_of_bbox(bbox):
    return int(bbox[3] - bbox[1])

def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])