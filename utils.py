import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_angle(start_point, end_point):
    vx = end_point[0] - start_point[0]
    vy = start_point[1] - end_point[1]  # Adjust for coordinate system
    pitch_radians = np.arctan2(vy, vx)
    pitch_degrees = np.degrees(pitch_radians)
    return pitch_degrees


def get_rotation(landmarks):
    average_four = (
        int((landmarks[8][0] + landmarks[12][0] + landmarks[16][0] + landmarks[20][0])/4),
        int((landmarks[8][1] + landmarks[12][1] + landmarks[16][1] + landmarks[20][1])/4),
    )
    angle_with_vertical = calculate_angle(landmarks[0], average_four)
    return angle_with_vertical, average_four


def calculate_average_distance(landmarks):
    key_indices = [0, 1, 2, 5, 9, 13, 17]
    distances = []
    for i in range(len(key_indices)):
        for j in range(i + 1, len(key_indices)):
            p1 = np.array([landmarks[key_indices[i]][0], landmarks[key_indices[i]][1]])
            p2 = np.array([landmarks[key_indices[j]][0], landmarks[key_indices[j]][1]])
            distance = np.linalg.norm(p1 - p2)
            distances.append(distance)
    return np.mean(distances)
