# The provided Python script conducts an experiment to evaluate the performance of the Single-Agent Bundle Adjustment (SABA) algorithm in recovering from artificially added noise to camera pose estimates
# and point estimates. The script allows for specifying various settings and parameters for the experiment, such as the type of noise, noise levels, smoothness parameters, visualization options, and saving figures.
# The results provide insights into how the algorithm performs under different noise conditions.

##################### Import modules and functions #####################
import sys
lib_location = "/home/pb/Documents/Thesis/Scripts/lib"
sys.path.insert(0, lib_location)

import numpy as np
from BA_functions import BA_noise
from functions import parsedata
import pandas as pd

################################            Load input data              ################################
ORB_version = 'ORB2'

keyframes_MH01, points_MH01 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH01.txt'.format(ORB_version))
keyframes_MH02, points_MH02 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH02.txt'.format(ORB_version))
keyframes_MH03, points_MH03 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH03.txt'.format(ORB_version))
keyframes_MH04, points_MH04 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH04.txt'.format(ORB_version))
keyframes_MH05, points_MH05 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH05.txt'.format(ORB_version))

################################             Apply functions               ################################

## Single-Agent
    # General settings
BA_algorithm = 'SABA'
BA_type = 'motion-only'
scenario = 'Scenario 1'

    # Generate sequence
sequence= 'MH01'
keyframes = [keyframes_MH01]
points = [points_MH01]
split_seq = [0, 0]

## Noise settings
noise_type = 'Gaussian'
thres_noise = 0    
noise_level = [np.array([0.01, 0.01, 0.01])]
noise_level = [0.03]

## Additional settings
smooth = 1
smooth_threshold = 1.6
visualize = 1
save = 0
save_fig_path = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Noise/{}/With ground truth/{}/{}/{}/'.format(BA_algorithm, sequence, noise_type, scenario)

# gt_path = '/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Ground Truth/state_estimate/CSV/gt_{}.csv'.format(ORB_version, sequence)
# ground_truth = pd.read_csv(gt_path)
ground_truth = 0
nbr_iter = 1
nbr_thread = -1

## Run function with settings defined above
APE_lim_tran, RPE_lim_tran, APE_unlim_tran, RPE_unlim_tran, \
APE_lim_rot, RPE_lim_rot, APE_unlim_rot, RPE_unlim_rot, \
nbr_iterations_req, cost_red, timings = \
BA_noise(nbr_thread, BA_algorithm, ground_truth, ORB_version, sequence, keyframes, points, smooth_threshold, split_seq, noise_type, BA_type, noise_level, nbr_iter, 1, thres_noise, visualize, save, save_fig_path)



# Print results
APE_noise_unlim_tran, APE_BA_unlim_tran, APE_smooth_unlim_tran = APE_unlim_tran[0], APE_unlim_tran[1], APE_unlim_tran[2]

APE_noise_lim_tran, APE_BA_lim_tran, APE_smooth_lim_tran  = APE_lim_tran[0], APE_lim_tran[1], APE_lim_tran[2]

APE_noise_unlim_rot, APE_BA_unlim_rot, APE_smooth_unlim_rot = APE_unlim_rot[0], APE_unlim_rot[1], APE_unlim_rot[2]

APE_noise_lim_rot, APE_BA_lim_rot, APE_smooth_lim_rot = APE_lim_rot[0], APE_lim_rot[1], APE_lim_rot[2]

RPE_noise_unlim_tran, RPE_BA_unlim_tran, RPE_smooth_unlim_tran = RPE_unlim_tran[0], RPE_unlim_tran[1], RPE_unlim_tran[2]

RPE_noise_lim_tran, RPE_BA_lim_tran, RPE_smooth_lim_tran = RPE_lim_tran[0], RPE_lim_tran[1], RPE_lim_tran[2]

RPE_noise_unlim_rot, RPE_BA_unlim_rot, RPE_smooth_unlim_rot = RPE_unlim_rot[0], RPE_unlim_rot[1], RPE_unlim_rot[2]

RPE_noise_lim_rot, RPE_BA_lim_rot, RPE_smooth_lim_rot = RPE_lim_rot[0], RPE_lim_rot[1], RPE_lim_rot[2]

print("\nAPE noise unlimited translation:", APE_noise_unlim_tran)
print("APE BA unlimited translation:", APE_BA_unlim_tran)
print("APE smooth unlimited translation:", APE_smooth_unlim_tran)

print("\nAPE noise unlimited rotation:", APE_noise_unlim_rot)
print("APE BA unlimited rotation:", APE_BA_unlim_rot)
print("APE smooth unlimited rotation:", APE_smooth_unlim_rot)

print("\nRPE noise unlimited translation:", RPE_noise_unlim_tran)
print("RPE BA unlimited translation:", RPE_BA_unlim_tran)
print("RPE smooth unlimited translation:", RPE_smooth_unlim_tran)

print("\nRPE noise unlimited rotation:", RPE_noise_unlim_rot)
print("RPE BA unlimited rotation:", RPE_BA_unlim_rot)
print("RPE smooth unlimited rotation:", RPE_smooth_unlim_rot)

print("\n\n")

print("\nAPE noise limited translation", APE_noise_lim_tran)
print("APE BA limited translation:", APE_BA_lim_tran)
print("APE smooth limited translation:", APE_smooth_lim_tran)

print("\nAPE noise limited rotation", APE_noise_lim_rot)
print("APE BA limited rotation:", APE_BA_lim_rot)
print("APE smooth limited rotation:", APE_smooth_lim_rot)

print("\nRPE noise limited translation", RPE_noise_lim_tran)
print("RPE BA limited translation:", RPE_BA_lim_tran)
print("RPE smooth limited translation:", RPE_smooth_lim_tran)

print("\nRPE noise limited rotation", RPE_noise_lim_rot)
print("RPE BA limited rotation:", RPE_BA_lim_rot)
print("RPE smooth limited rotation:", RPE_smooth_lim_rot)
