import numpy as np
import csv, datetime, os
from exp_article_test.uav import UAV

def get_rela_angle(arr1, arr2):
    if np.linalg.norm(arr1)==0 or np.linalg.norm(arr2)==0:
        return 0
    theta = arr1.dot(arr2)/(np.linalg.norm(arr1)*np.linalg.norm(arr2)+1e-6)
    theta = np.clip(theta, -1.0, 1.0)
    return np.arccos(theta)

def create_directory(path: str, sub_path_list: list):
    for sub_path in sub_path_list:
        if not os.path.exists(path + sub_path):
            os.makedirs(path + sub_path, exist_ok=True)
            print('Path: {} create successfully!'.format(path + sub_path))
        else:
            print('Path: {} is already existence!'.format(path + sub_path))

def get_time_string():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%dd_%Hh_%Mm_%Ss")
    return formatted_datetime

def save_uav_csv(uav : UAV, filepath):
    filename = f"f16_[{uav.team}]_{uav.id}_{uav.info}.csv"
    with open(filepath + filename, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time", "x", "y", "z",
                                 "Roll(rad)", "Pitch(rad)", "Yaw(rad)",
                                 "airspeed", "Blood"])
        for i in range(len(uav.record_t)):
            writer.writerow([uav.record_t[i], uav.record_pos[i][0], uav.record_pos[i][1], uav.record_pos[i][2],
                             uav.record_phi[i], uav.record_theta[i], uav.record_psi[i],
                             uav.record_airspeed[i], uav.record_blood[i]])