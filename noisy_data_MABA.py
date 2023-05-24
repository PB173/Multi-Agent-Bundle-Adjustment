# The goal of this experiment is to show how the Bundle Adjustment algorithm recovers from artificially added noise to the camera pose estimates and the point estimates.

##################### Import modules and functions #####################
import sys
lib_location = "/home/pb/Documents/Thesis/Scripts/lib"
sys.path.insert(0, lib_location)

import numpy as np
import pickle
from BA_functions import BA_noise
from functions import parsedata


################################            Load input data              ################################
ORB_version = 'ORB2'

keyframes_MH01, points_MH01 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH01.txt'.format(ORB_version))
keyframes_MH02, points_MH02 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH02.txt'.format(ORB_version))
keyframes_MH03, points_MH03 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH03.txt'.format(ORB_version))
keyframes_MH04, points_MH04 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH04.txt'.format(ORB_version))
keyframes_MH05, points_MH05 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH05b.txt'.format(ORB_version))


################################             Apply functions               ################################

## Multi-Agent
    # General settings
BA_algorithm = 'MABA'
BA_type = 'complete'
scenario = 'Scenario 1'

    # Load sequences
path =  '/home/pb/Documents/Thesis/Data/MABA/Combined data/' # Load common data structs
sequence = 'MH01_MH02'

with open(path + 'keyframes_combi_{}.pkl'.format(sequence), 'rb') as f:
    keyframes_combi = pickle.load(f)

with open(path + 'points_combi_{}.pkl'.format(sequence), 'rb') as f:
    points_combi = pickle.load(f)

# for point in points_combi:
#     for p in point:
#         for test in p:
#             print(test)
            
#         print(" ")
#     print(" ")


split_seq = [0, len(keyframes_MH01), len(keyframes_combi[0])]

## Noise settings
noise_type = 'Random Walk'
thres_noise = 0    
noise_level = [0.015]

## Additional settings
smooth = 1
smooth_threshold = 1.15
visualize = 1
save = 0
save_fig_path = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Noise/{}/{}/{}/{}/'.format(BA_algorithm, sequence, noise_type, scenario)

ground_truth = 0
nbr_iter = 1
nbr_thread = -1

## Run function with settings defined above
APE_lim_tran, RPE_lim_tran, APE_unlim_tran, RPE_unlim_tran, \
APE_lim_rot, RPE_lim_rot, APE_unlim_rot, RPE_unlim_rot, \
nbr_iterations_req, cost_red, timings = \
BA_noise(nbr_thread, BA_algorithm, ground_truth, ORB_version, sequence, keyframes_combi, points_combi, smooth_threshold, split_seq, noise_type, BA_type, noise_level, nbr_iter, 1, thres_noise, visualize, save, save_fig_path)

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
