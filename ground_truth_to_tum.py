from evo.tools import file_interface
import numpy as np
import pandas as pd
import sys
lib_location = "/home/pb/Documents/Thesis/Scripts/lib"
sys.path.insert(0, lib_location)

from project_keyframe import inverse_transformation_matrix
from rotation_transformations import quat2rotmat, rotmat2quat

ORB_version = 'ORB2'
gt_data_path = '/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Ground Truth/state_estimate/'.format(ORB_version)

# Function to perform transformation and create csv file
def transform2csv(sequence, traj, quat, T_SB):
    gt_tran, gt_quat = [], []
    for i in range(len(traj)):
        R = np.reshape(quat2rotmat(quat[i], "vector"), (3,3))
        print(R)
        t = np.reshape(traj[i], (3,1))
        T = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
        T_tf = np.matmul(T_SB, T)
        gt_t = np.array(np.reshape(T_tf[:3,3],(-1,3)))[0] # extract translation vector
        gt_R = np.reshape(T_tf[:3,:3], (-1,9))[0] # extract rotation vector
        print("T tf:", T_tf)
        print(np.reshape(gt_R, (3,3)))
        gt_q = rotmat2quat(gt_R, "wxyz")

        gt_tran.append(gt_t), gt_quat.append(gt_q)

        # Push data to csv file
    df = pd.read_csv(gt_data_path + 'CSV/gt_{}_transform.csv'.format(sequence))

    df[' p_RS_R_x [m]'] = [gt_tr[0] for gt_tr in gt_tran]
    df[' p_RS_R_y [m]'] = [gt_tr[1] for gt_tr in gt_tran]
    df[' p_RS_R_z [m]'] = [gt_tr[2] for gt_tr in gt_tran]

    df[' q_RS_w []'] = [gt_q[0] for gt_q in gt_quat]
    df[' q_RS_x []'] = [gt_q[1] for gt_q in gt_quat]
    df[' q_RS_y []'] = [gt_q[2] for gt_q in gt_quat]
    df[' q_RS_z []'] = [gt_q[3] for gt_q in gt_quat]

    df.to_csv(gt_data_path + 'CSV/gt_{}_transform.csv'.format(sequence), index=False)


# Convert keyframes and points to body frame using T_BS provided in sensor yaml file
    # Invert matrix to have proper transformation
T_BS = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
    [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
    [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
    [0.0, 0.0, 0.0, 1.0]])

R_BS = T_BS[:3,:3]
t_BS = T_BS[:3, 3]

T_SB = np.reshape(inverse_transformation_matrix(R_BS, t_BS, "vector"), (4,4))

    
    # Read ground truth camera poses
gt_MH01 = pd.read_csv('/home/pb/Documents/EUROC dataset/ASL/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv',
                      usecols=[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]',' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []'])
gt_MH02 = pd.read_csv('/home/pb/Documents/EUROC dataset/ASL/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv',
                      usecols=[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]',' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []'])
gt_MH03 = pd.read_csv('/home/pb/Documents/EUROC dataset/ASL/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv',
                      usecols=[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]',' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []'])
gt_MH04 = pd.read_csv('/home/pb/Documents/EUROC dataset/ASL/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv',
                      usecols=[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]',' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []'])
gt_MH05 = pd.read_csv('/home/pb/Documents/EUROC dataset/ASL/MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv',
                      usecols=[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]',' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []'])

    # Apply transformation to data
traj_MH01 = [[gt_MH01[' p_RS_R_x [m]'][i], gt_MH01[' p_RS_R_y [m]'][i], gt_MH01[' p_RS_R_z [m]'][i]] for i in range(len(gt_MH01[' p_RS_R_x [m]']))]
quat_MH01 = [[gt_MH01[' q_RS_w []'][i], gt_MH01[' q_RS_x []'][i], gt_MH01[' q_RS_y []'][i], gt_MH01[' q_RS_z []'][i]] for i in range(len(gt_MH01[' q_RS_w []']))]

traj_MH02 = [[gt_MH02[' p_RS_R_x [m]'][i], gt_MH02[' p_RS_R_y [m]'][i], gt_MH02[' p_RS_R_z [m]'][i]] for i in range(len(gt_MH02[' p_RS_R_x [m]']))]
quat_MH02 = [[gt_MH02[' q_RS_w []'][i], gt_MH02[' q_RS_x []'][i], gt_MH02[' q_RS_y []'][i], gt_MH02[' q_RS_z []'][i]] for i in range(len(gt_MH02[' q_RS_w []']))]

traj_MH03 = [[gt_MH03[' p_RS_R_x [m]'][i], gt_MH03[' p_RS_R_y [m]'][i], gt_MH03[' p_RS_R_z [m]'][i]] for i in range(len(gt_MH03[' p_RS_R_x [m]']))]
quat_MH03 = [[gt_MH03[' q_RS_w []'][i], gt_MH03[' q_RS_x []'][i], gt_MH03[' q_RS_y []'][i], gt_MH03[' q_RS_z []'][i]] for i in range(len(gt_MH03[' q_RS_w []']))]

traj_MH04 = [[gt_MH04[' p_RS_R_x [m]'][i], gt_MH04[' p_RS_R_y [m]'][i], gt_MH04[' p_RS_R_z [m]'][i]] for i in range(len(gt_MH04[' p_RS_R_x [m]']))]
quat_MH04 = [[gt_MH04[' q_RS_w []'][i], gt_MH04[' q_RS_x []'][i], gt_MH04[' q_RS_y []'][i], gt_MH04[' q_RS_z []'][i]] for i in range(len(gt_MH04[' q_RS_w []']))]

traj_MH05 = [[gt_MH05[' p_RS_R_x [m]'][i], gt_MH05[' p_RS_R_y [m]'][i], gt_MH05[' p_RS_R_z [m]'][i]] for i in range(len(gt_MH05[' p_RS_R_x [m]']))]
quat_MH05 = [[gt_MH05[' q_RS_w []'][i], gt_MH05[' q_RS_x []'][i], gt_MH05[' q_RS_y []'][i], gt_MH05[' q_RS_z []'][i]] for i in range(len(gt_MH05[' q_RS_w []']))]

    # Create csv files
transform2csv('MH01', traj_MH01[:5], quat_MH01, T_SB)
# transform2csv('MH02', traj_MH02, quat_MH02, T_SB)
# transform2csv('MH03', traj_MH03, quat_MH03, T_SB)
# transform2csv('MH04', traj_MH04, quat_MH04, T_SB)
# transform2csv('MH05', traj_MH05, quat_MH05, T_SB)

    # Create TUM files

# gt_data_MH01 = file_interface.read_euroc_csv_trajectory(gt_data_path + 'CSV/gt_MH01_transform.csv')
# gt_data_MH02 = file_interface.read_euroc_csv_trajectory(gt_data_path + 'CSV/gt_MH02_transform.csv')
# gt_data_MH03 = file_interface.read_euroc_csv_trajectory(gt_data_path + 'CSV/gt_MH03_transform.csv')
# gt_data_MH04 = file_interface.read_euroc_csv_trajectory(gt_data_path + 'CSV/gt_MH04_transform.csv')
# gt_data_MH05 = file_interface.read_euroc_csv_trajectory(gt_data_path + 'CSV/gt_MH05_transform.csv')

# file_interface.write_tum_trajectory_file(gt_data_path + 'TUM/gt_MH01_TUM_transform.txt', gt_data_MH01)
# file_interface.write_tum_trajectory_file(gt_data_path + 'TUM/gt_MH02_TUM_transformtxt', gt_data_MH02)
# file_interface.write_tum_trajectory_file(gt_data_path + 'TUM/gt_MH03_TUM_transform.txt', gt_data_MH03)
# file_interface.write_tum_trajectory_file(gt_data_path + 'TUM/gt_MH04_TUM_transform.txt', gt_data_MH04)
# file_interface.write_tum_trajectory_file(gt_data_path + 'TUM/gt_MH05_TUM_transform.txt', gt_data_MH05)


# # Create CSV data
# gt_data_MH01 = file_interface.read_euroc_csv_trajectory(gt_data_path + 'CSV/gt_MH01.csv')
# gt_data_MH02 = file_interface.read_euroc_csv_trajectory(gt_data_path + 'CSV/gt_MH02.csv')
# gt_data_MH03 = file_interface.read_euroc_csv_trajectory(gt_data_path + 'CSV/gt_MH03.csv')
# gt_data_MH04 = file_interface.read_euroc_csv_trajectory(gt_data_path + 'CSV/gt_MH04.csv')
# gt_data_MH05 = file_interface.read_euroc_csv_trajectory(gt_data_path + 'CSV/gt_MH05.csv')

# # Create TUM file
# file_interface.write_tum_trajectory_file(gt_data_path + 'TUM/gt_MH01_TUM.txt', gt_data_MH01)
# file_interface.write_tum_trajectory_file(gt_data_path + 'TUM/gt_MH02_TUM.txt', gt_data_MH02)
# file_interface.write_tum_trajectory_file(gt_data_path + 'TUM/gt_MH03_TUM.txt', gt_data_MH03)
# file_interface.write_tum_trajectory_file(gt_data_path + 'TUM/gt_MH04_TUM.txt', gt_data_MH04)
# file_interface.write_tum_trajectory_file(gt_data_path + 'TUM/gt_MH05_TUM.txt', gt_data_MH05)
