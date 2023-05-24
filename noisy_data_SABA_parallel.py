##################### Import modules and functions #####################
import sys
lib_location = "/home/pb/Documents/Thesis/Scripts/lib"
sys.path.insert(0, lib_location)

import numpy as np
from BA_functions import BA_noise, BA_noise_parallel
from functions import parsedata


################################            Load input data              ################################
ORB_version = 'ORB2'

keyframes_MH01, points_MH01 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH01.txt'.format(ORB_version))
keyframes_MH02, points_MH02 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH02.txt'.format(ORB_version))
keyframes_MH03, points_MH03 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH03.txt'.format(ORB_version))
keyframes_MH04, points_MH04 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH04.txt'.format(ORB_version))
keyframes_MH05, points_MH05 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH05.txt'.format(ORB_version))
keyframes_MH05, points_MH05 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Distorted keypoints/MH05b.txt'.format(ORB_version))


################################             Apply functions               ################################

# If MutEx does not work, maybe try to give every sequence it's own individual path for storing and reading the TUM data => avoid race condition!

nbr_thread, BA_algorithm, ORB_version, sequence = range(11), 'SABA', 'ORB2', 'MH02'
keyframes, points, smooth_threshold, split_seq = [keyframes_MH02], [points_MH02], 1.1, [0, 0]
noise_type, BA_type, noise_level, nbr_iter = 'Random Walk', 'complete', np.linspace(0, 0.15, 11), 50
th_nbr_kf, thres_noise, visualize, save, ground_truth = 1, 0, 0, 0, 0
scenario = ''
save_fig_path = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Analysis/Noise/{}/{}/{}/'.format(sequence, noise_type, scenario)

args = [nbr_thread, BA_algorithm, ground_truth, ORB_version, sequence, keyframes, points, smooth_threshold, split_seq, noise_type, BA_type, \
        noise_level, nbr_iter, th_nbr_kf, thres_noise, visualize, save, save_fig_path]

APE_lim_tran, RPE_lim_tran, APE_unlim_tran, RPE_unlim_tran, APE_lim_rot, RPE_lim_rot, \
APE_unlim_rot, RPE_unlim_rot, iterations, cost_reduction, timings = BA_noise_parallel(args)

## Save results
path = '/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 6/{}/'.format(sequence)
np.save(path + 'APE_unlim_tran_{}.npy'.format(sequence), APE_unlim_tran)
np.save(path + 'APE_unlim_rot_{}.npy'.format(sequence), APE_unlim_rot)
np.save(path + 'RPE_unlim_tran_{}.npy'.format(sequence), RPE_unlim_tran)
np.save(path + 'RPE_unlim_rot_{}.npy'.format(sequence), RPE_unlim_rot)

np.save(path + 'APE_lim_tran_{}.npy'.format(sequence), APE_lim_tran)
np.save(path + 'APE_lim_rot_{}.npy'.format(sequence), APE_lim_rot)
np.save(path + 'RPE_lim_tran_{}.npy'.format(sequence), RPE_lim_tran)
np.save(path + 'RPE_lim_rot_{}.npy'.format(sequence), RPE_lim_rot)

np.save(path + 'nbr_iterations_{}.npy'.format(sequence), iterations)
np.save(path + 'cost_red_{}.npy'.format(sequence), cost_reduction)
np.save(path + 'timings_{}.npy'.format(sequence), timings)

# np.save(path + 'nbr_iterations_unlim_{}.npy'.format(sequence), iterations)
# np.save(path + 'cost_red_unlim_{}.npy'.format(sequence), cost_reduction)
# np.save(path + 'timings_unlim_{}.npy'.format(sequence), timings)