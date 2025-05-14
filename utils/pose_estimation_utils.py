import math

def calculate_angle(a, b, c):
    '''
    Calculate the angle between three points a, b, and c using the law of cosines.
    The angle is calculated at point b, between the lines ab and bc.
    The points are represented as tuples (x, y).
    The angle is returned in degrees.
    The function uses the law of cosines to calculate the angle:
    cos(angle) = dot_product(a, c) / (|a| * |c|)
    angle = cos^-1(dot_product(a, c) / (|a| * |c|))
    '''
    # calculate the distance between a and b
    ba = [a['x'] - b['x'], a['y'] - b['y']]
    bc = [c['x'] - b['x'], c['y'] - b['y']]

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]

    # magnitude of (a, b) and (c, b)
    mag_a = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_c = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_a == 0 or mag_c == 0:
        return 0
    # calculate the angle in radians
    angle = math.acos(dot_product / (mag_a * mag_c))
    # convert the angle to degrees
    angle = math.degrees(angle)

    return angle

def get_closest_kp_to_player(point, kps, kps_idxs):
    closest_dist = float('inf')
    kp_idx = kps_idxs[0]

    for idx in kps_idxs:
        kp = (kps[idx * 2], kps[idx * 2 + 1])
        dist = math.sqrt((point[0] - kp[0]) ** 2 + (point[1] - kp[1]) ** 2)
        
        if dist < closest_dist:
            closest_dist = dist
            kp_idx = idx

    return kp_idx