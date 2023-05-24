from Visualizations.visualizations import compare_trajectories, compare_noisy_trajectories, plot_noisy_pointclouds
from functions import csv2tum, save_kfBA2csv, save_pointsBA2csv, save_noisy_kf2csv, readfromTUM, information_keyframes, saveARPE2csv
from project_keyframe import project_noisy_data_to_keyframe
from bundle_adjustment import BundleAdjustment_SA, smoothen_trajectory
from Evo.evo_metrics import visualize_aligned_keyframes, visualize_results_aligned, calculate_APE_simple, calculate_RPE_simple, calculate_APE

import numpy as np
import copy
import multiprocessing as mp
from evo.core import sync, trajectory

def apply_BundleAdjustment(BA_algorithm, ground_truth, ORB_version, sequence, keyframes, points, smooth_threshold, split_seq, T_BS, th_nbr_kf, cam_parameters, noise_type, noise_factor, threshold_noise, visualize_BA = 1, visualize_reprojection=0, directory='', BA_type = '', limit = '', smooth = 0, verbose='output', max_iter=200, plot_cost=0, save=0, save_fig_path='', path = '', save_data=0):
    """
    Apply Bundle Adjustment to a sequence of keyframes and points.

    Parameters:
        - BA_algorithm (str): The algorithm used for Bundle Adjustment.
        - ground_truth: Ground truth data for comparison.
        - ORB_version: Version of the ORB algorithm used (options are 'ORB2' or 'ORB3').
        - sequence (str): Name of the sequence.
        - keyframes (list): List of keyframes.
        - points (list): List of corresponding points.
        - smooth_threshold: Threshold for smoothing the results.
        - split_seq : Indicates the length of the first sequence when the MABA algorithm is considered.
        - T_BS: Transformation matrix from sensor to body frame.
        - th_nbr_kf (int): Threshold number of keyframes. Points seen by fewer keyframes than this threshold are discarded.
        - cam_parameters: Camera parameters ([fx, fy, cx, cy]).
        - noise_type: Type of noise (options are 'Gaussian, 'Uniform' or 'Random Walk').
        - noise_factor: Noise factor.
        - threshold_noise: Threshold for applying noise.
        - visualize_BA (int): Flag to visualize Bundle Adjustment results.
        - visualize_reprojection (int): Flag to visualize reprojection results.
        - directory (str): Directory for saving files.
        - BA_type (str): Type of Bundle Adjustment.
        - limit (str): Limit information.
        - smooth (int): Flag to enable smoothing.
        - verbose (str): Verbosity level.
        - max_iter (int): Maximum number of iterations for Bundle Adjustment.
        - plot_cost (int): Flag to plot the cost during Bundle Adjustment.
        - save (int): Flag to save files.
        - save_fig_path (str): File path for saving figures.
        - path (str): Path information.
        - save_data (int): Flag to save data.

    Returns:
        - all_org_data: List of original data.
        - all_noisy_data: List of noisy data.
        - all_results_BA: List of Bundle Adjustment results.
        - all_sol_stats: List of solution statistics.
        - smooth_data: Smoothed trajectory data.

    """
    # Initialize vectors
    all_org_data = []
    all_noisy_data = []
    all_results_BA = []
    all_sol_stats = []

    # Loop over all keyframes
    for i in range(len(keyframes)):
        kf_seq = keyframes[i]
        pts_seq = points[i]

        org_data, noisy_data, BA_data, _, _, sol_stats = BundleAdjustment_SA(ORB_version, kf_seq, pts_seq, T_BS, th_nbr_kf, cam_parameters, noise_type, noise_factor, threshold_noise, BA_type, max_iter, 'full', plot_cost, save, save_fig_path, verbose, limit, BA_algorithm, split_seq[1]) 


        all_org_data.append(org_data)
        all_noisy_data.append(noisy_data)
        all_results_BA.append(BA_data)
        all_sol_stats.append(sol_stats)

        # Smoothen the results obtained with the BA to reject obvious outliers
        if smooth: 
            smooth_data = smoothen_trajectory(BA_data[0], 0.5, smooth_threshold) # resulting trajectory after applying BA   

        
        # Optionally: visualize results of Bundle Adjustment (as 1 large sequence for MABA)
        if visualize_BA:
            if BA_algorithm == 'SABA':
                sequence_plot = sequence
            elif BA_algorithm == 'MABA':
                sequence_plot =  '{} & {}'.format(sequence.split('_')[0],sequence.split('_')[1] )
            if np.all(noise_factor==0):
                compare_trajectories(kf_seq, BA_data[0], ORB_version, idx= i+1, save=save) 
            else:
                timestamps_kf = information_keyframes(kf_seq)[1]
                compare_noisy_trajectories(org_data[0], noisy_data[0], BA_data[0], smooth_data, ground_truth, timestamps_kf, sequence_plot, save, save_fig_path)
                plot_noisy_pointclouds(org_data[1][:], noisy_data[1][:], BA_data[1][:], sequence_plot, save, save_fig_path)
        
        # Save results of Bundle Adjustment as csv file
        if save_data == 1 and BA_algorithm == 'SABA':
            # print("Saving data for SABA")
            save_kfBA2csv(path + 'CSV/BA_kf_{}.csv'.format(sequence), kf_seq,  BA_data[0], BA_data[2],"wxyz")
            save_pointsBA2csv(path + 'CSV/BA_points_{}.csv'.format(sequence), BA_data[1])
            if np.any(noise_factor): # Store noisy data
                save_noisy_kf2csv(path + 'CSV/Noisy_kf_{}.csv'.format(sequence), kf_seq, noisy_data[0], noisy_data[2], "wxyz")
            save_noisy_kf2csv(path + 'CSV/Noisy_kf_{}.csv'.format(sequence), kf_seq, noisy_data[0], noisy_data[2], "wxyz")
            if smooth:
                save_noisy_kf2csv(path + 'CSV/Smooth_kf_{}.csv'.format(sequence), kf_seq, smooth_data, BA_data[2],"wxyz")
            
            save_noisy_kf2csv(path + 'CSV/Org_kf_{}.csv'.format(sequence), kf_seq, org_data[0], org_data[2],"wxyz") # Save original data as csv

        # Split results back in original sequences before saving: original, noisy, BA
        if save_data == 1 and BA_algorithm == 'MABA':
            # print("Saving data for MABA")
            for i in range(1, len(split_seq)): # start at 1, must insert 0 as first element and total length as final!
                keyframes = kf_seq[split_seq[i-1]:split_seq[i]]

                traj_org = org_data[0][split_seq[i-1]:split_seq[i]]
                rot_org  = org_data[2][split_seq[i-1]:split_seq[i]]

                traj_BA = BA_data[0][split_seq[i-1]:split_seq[i]]
                points_BA =  BA_data[1][split_seq[i-1]:split_seq[i]]
                rot_BA  = BA_data[2][split_seq[i-1]:split_seq[i]]

                save_kfBA2csv(path + 'CSV/Org_kf_{}_part_{}.csv'.format(sequence,i), keyframes,  traj_org, rot_org, "wxyz") # Save original data as csv

                save_kfBA2csv(path + 'CSV/BA_kf_{}_part_{}.csv'.format(sequence,i), keyframes,  traj_BA, rot_BA, "wxyz") # Store data after BA
                save_pointsBA2csv(path + 'CSV/BA_points_{}_part_{}.csv'.format(sequence, i), points_BA)
                
                if np.any(noise_factor): # Store noisy data
                    pass
                traj_noisy = noisy_data[0][split_seq[i-1]:split_seq[i]]
                rot_noisy = noisy_data[2][split_seq[i-1]:split_seq[i]]
                save_noisy_kf2csv(path + 'CSV/Noisy_kf_{}_part_{}.csv'.format(sequence, i), keyframes, traj_noisy, rot_noisy, "wxyz")

                if smooth:
                    traj_smooth = smooth_data[split_seq[i-1]:split_seq[i]]
                    save_noisy_kf2csv(path + 'CSV/Smooth_kf_{}_part_{}.csv'.format(sequence, i), keyframes, traj_smooth, rot_BA,"wxyz")

        if visualize_reprojection == 1 and BA_algorithm == 'SABA':
            project_noisy_data_to_keyframe(directory, ORB_version, sequence, kf_seq, pts_seq, noisy_data[1], noisy_data[0], cam_parameters, BA_data, after_BA = 1, visualize = 1, save_fig=1, save_fig_path=save_fig_path, save_to_gif=0)
   
    return all_org_data, all_noisy_data, all_results_BA, all_sol_stats, smooth_data


def initialize_directories(BA_algorithm, ORB_version, sequence, nbr_thread=-1):
    if BA_algorithm == 'SABA':
        if sequence == 'MH01' or sequence == 'MH02':
            sequence_dir = 'MH_0' + sequence[-1] + '_easy'
        elif sequence == 'MH03':
            sequence_dir = 'MH_03_medium'
        elif sequence == 'MH04' or sequence == 'MH05':
            sequence_dir = 'MH_0' + sequence[-1] + '_difficult'
        
        if nbr_thread == -1:
            path = '/home/pb/Documents/Thesis/Data/SABA/{}/Noise/'.format(ORB_version)
        elif nbr_thread >=0:
            path = '/home/pb/Documents/Thesis/Data/Sensitivity analysis/Influence sigma/MO/Scenario 6/{}/Data/Thread {}/'.format(sequence, nbr_thread)
        dir = '/home/pb/Documents/EUROC dataset/ASL/{}/mav0/cam0/data/'.format(sequence_dir)
    else:
        if nbr_thread == -1:
            path = '/home/pb/Documents/Thesis/Data/MABA/{}/Noise/'.format(ORB_version)
        elif nbr_thread >=0:
            path = '/home/pb/Documents/Thesis/Data/Sensitivity analysis/MABA/Scenario 6/{}/Data/Thread {}/'.format(sequence, nbr_thread)
        dir = ''

    return path, dir

def BA_noise(nbr_thread, BA_algorithm, ground_truth, ORB_version, sequence, keyframes, points, smooth_threshold, split_seq, noise_type, BA_type, noise_level, nbr_iter, th_nbr_kf, thres_noise = 10, visualize = 0, save =0, save_fig_path_temp =''):

    # Initialize all paths and directories
    path, dir = initialize_directories(BA_algorithm, ORB_version, sequence, nbr_thread)
    
    # Initialize camera parameters
    cam_intrinsics = np.array([458.654, 457.296, 367.215, 248.375]) # Values from ORB_SLAM2 EuRoC.yaml file ([fx, fy, cx, cy]) 
    distortion_coef = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]) # [k1, k2, P1, P2]
    cam_parameters = np.concatenate((cam_intrinsics, distortion_coef), axis=None)

    T_BS = 0
    
    # Run 4 different scenarios:
    APE_rmse_noise_lim_tran, APE_rmse_BA_lim_tran, APE_rmse_smooth_lim_tran = [], [], []
    APE_rmse_noise_unlim_tran, APE_rmse_BA_unlim_tran, APE_rmse_smooth_unlim_tran = [], [], []
    RPE_rmse_noise_lim_tran, RPE_rmse_BA_lim_tran, RPE_rmse_smooth_lim_tran = [], [], []
    RPE_rmse_noise_unlim_tran, RPE_rmse_BA_unlim_tran, RPE_rmse_smooth_unlim_tran = [], [], []

    APE_rmse_noise_lim_rot, APE_rmse_BA_lim_rot, APE_rmse_smooth_lim_rot = [], [], []
    APE_rmse_noise_unlim_rot, APE_rmse_BA_unlim_rot, APE_rmse_smooth_unlim_rot = [], [], []
    RPE_rmse_noise_lim_rot, RPE_rmse_BA_lim_rot, RPE_rmse_smooth_lim_rot = [], [], []
    RPE_rmse_noise_unlim_rot, RPE_rmse_BA_unlim_rot, RPE_rmse_smooth_unlim_rot = [], [], []

    iterations, cost_reduction, timings = [], [], []                    

    for nl in noise_level:
        for limit in ['No limits','Limited']:
        # for limit in ['Limited']:
        # for limit in ['No limits']:
            save_fig_path = save_fig_path_temp + limit +'/'


            # Iterate multiple times to calculate statistics
            iter, cost_red, time = [], [], []
            for i in range(nbr_iter):
                print('sig = {}, {}, iteration = {}'.format(nl, limit, i))

                _, _, _, sol_stats, _ = apply_BundleAdjustment(BA_algorithm, ground_truth, ORB_version, sequence, keyframes, points, smooth_threshold, split_seq, T_BS, th_nbr_kf, \
                                                               cam_parameters, noise_type, nl, thres_noise, visualize, 0, dir, BA_type, limit, 1, 'output', 500, visualize, save, save_fig_path, path, save_data=1)
                iter.append(sol_stats[0][0])
                cost_red.append(sol_stats[0][1])
                time.append(sol_stats[0][2])
                
                # SABA: read output files
                if BA_algorithm == 'SABA':
                    seq = sequence
                    # Alignment with the ground truth 
                        # Convert keyframes from CSV to TUM
                    csv2tum(path + 'CSV/Org_kf_{}.csv'.format(sequence), path + 'TUM/Org_kf_{}_TUM.txt'.format(sequence)) # Original input data
                    csv2tum(path + 'CSV/Noisy_kf_{}.csv'.format(sequence), path + 'TUM/Noisy_kf_{}_TUM.txt'.format(sequence)) # Noisy input data
                    csv2tum(path + 'CSV/BA_kf_{}.csv'.format(sequence), path + 'TUM/BA_kf_{}_TUM.txt'.format(sequence)) # Results of BA
                    csv2tum(path + 'CSV/Smooth_kf_{}.csv'.format(sequence), path + 'TUM/Smooth_kf_{}_TUM.txt'.format(sequence)) # Results of BA (smooth)

                        # Extract TUM data and ground truth to align
                    org_path, noise_path, BA_path, smooth_path, gt_path = [], [], [], [], []

                    org_path.append(path + 'TUM/Org_kf_{}_TUM.txt'.format(sequence))
                    noise_path.append(path + 'TUM/Noisy_kf_{}_TUM.txt'.format(sequence))
                    BA_path.append(path + 'TUM/BA_kf_{}_TUM.txt'.format(sequence))
                    smooth_path.append(path + 'TUM/Smooth_kf_{}_TUM.txt'.format(sequence))
                    gt_path.append('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Ground Truth/state_estimate/TUM/gt_{}_TUM.txt'.format(ORB_version, sequence))
                    # gt_path = '/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Ground Truth/state_estimate/TUM/gt_{}_TUM_transform.txt'.format(ORB_version, sequence)

                    traj_org, traj_noise, traj_BA, traj_smooth, traj_gt = [], [], [], [], []

                    traj_org.append(readfromTUM(org_path[-1]))
                    traj_noise.append(readfromTUM(noise_path[-1]))
                    traj_BA.append(readfromTUM(BA_path[-1]))
                    traj_smooth.append(readfromTUM(smooth_path[-1]))
                    traj_gt.append(readfromTUM(gt_path[-1]))

                # In case of MABA, read different sequences
                if BA_algorithm == 'MABA':
                    # seq = sequence.split('_')[i]
                    org_path, noise_path, BA_path, smooth_path, gt_path = [], [], [], [], []
                    traj_org, traj_noise, traj_BA, traj_smooth, traj_gt = [], [], [], [], []
                    for j in range(len(split_seq)-1):

                        # Alignment with the ground truth 
                            # Convert keyframes from CSV to TUM
                        csv2tum(path + 'CSV/Org_kf_{}_part_{}.csv'.format(sequence, j+1),
                                path + 'TUM/Org_kf_{}_TUM_part_{}.txt'.format(sequence, j+1)) # Original input data                            
                        csv2tum(path + 'CSV/Noisy_kf_{}_part_{}.csv'.format(sequence, j+1),
                                path + 'TUM/Noisy_kf_{}_TUM_part_{}.txt'.format(sequence, j+1)) # Noisy input data
                        csv2tum(path + 'CSV/BA_kf_{}_part_{}.csv'.format(sequence, j+1),
                                path + 'TUM/BA_kf_{}_TUM_part_{}.txt'.format(sequence, j+1)) # Results of BA
                        csv2tum(path + 'CSV/Smooth_kf_{}_part_{}.csv'.format(sequence, j+1),
                                path + 'TUM/Smooth_kf_{}_TUM_part_{}.txt'.format(sequence, j+1)) # Results of BA (smooth)

                            # Extract TUM data  and ground truth to align
                        org_path.append(path + 'TUM/Org_kf_{}_TUM_part_{}.txt'.format(sequence, j+1))
                        noise_path.append(path + 'TUM/Noisy_kf_{}_TUM_part_{}.txt'.format(sequence, j+1))
                        BA_path.append(path + 'TUM/BA_kf_{}_TUM_part_{}.txt'.format(sequence, j+1))
                        smooth_path.append(path + 'TUM/Smooth_kf_{}_TUM_part_{}.txt'.format(sequence, j+1))
                        gt_path.append('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Ground Truth/state_estimate/TUM/gt_{}_TUM.txt'.format(ORB_version, sequence.split('_')[j]))

                        traj_org.append(readfromTUM(org_path[-1]))
                        traj_noise.append(readfromTUM(noise_path[-1]))
                        traj_BA.append(readfromTUM(BA_path[-1]))
                        traj_smooth.append(readfromTUM(smooth_path[-1]))
                        traj_gt.append(readfromTUM(gt_path[-1]))


                # Perform association based on timestamps
                max_diff = 0.01
                for i in range(len(traj_noise)): # For MABA, execute code for every trajectory individually (for SABA, len=1 so no change)
                    traj_gt_ass_org, traj_org_ass = sync.associate_trajectories(traj_gt[i], traj_org[i], max_diff)
                    traj_gt_ass_noise, traj_noise_ass = sync.associate_trajectories(traj_gt[i], traj_noise[i], max_diff)
                    traj_gt_ass_BA, traj_BA_ass = sync.associate_trajectories(traj_gt[i], traj_BA[i], max_diff)
                    traj_gt_ass_smooth, traj_smooth_ass = sync.associate_trajectories(traj_gt[i], traj_smooth[i], max_diff)

                    
                        # Align estimates and ground truth and correct scale (for monocular images)
                    print("Aligning the original trajectory...")
                    traj_org_ass_aligned = copy.deepcopy(traj_org_ass)
                    traj_org_ass_aligned.align(traj_gt_ass_org, correct_scale=True, correct_only_scale=False)

                    print("Aligning the noisy trajectory...")
                    traj_noise_ass_aligned = copy.deepcopy(traj_noise_ass)
                    traj_noise_ass_aligned.align(traj_gt_ass_noise, correct_scale=True, correct_only_scale=False)

                    print("Aligning the BA trajectory...")
                    traj_BA_ass_aligned = copy.deepcopy(traj_BA_ass)
                    traj_BA_ass_aligned.align(traj_gt_ass_BA, correct_scale=True, correct_only_scale=False)

                    print("Aligning the smoothed trajectory...")
                    traj_smooth_ass_aligned = copy.deepcopy(traj_smooth_ass)
                    traj_smooth_ass_aligned.align(traj_gt_ass_smooth, correct_scale=True, correct_only_scale=False)


                        # Visualize alignment

                    if BA_algorithm == 'MABA':
                        seq = sequence.split('_')[i]
                        save_figure_path = save_fig_path + seq + '/'
                    elif BA_algorithm == 'SABA':
                        save_figure_path = save_fig_path
                    
                    if visualize:
                        trajectories = [traj_org_ass_aligned, traj_noise_ass_aligned, traj_BA_ass_aligned, traj_smooth_ass_aligned, traj_gt_ass_org]
                        visualize_results_aligned(trajectories, save, save_figure_path)

                        # visualize_aligned_keyframes(traj_noise_ass_aligned, traj_gt_ass_noise, seq, 'Alignment of the camera trajectory: {}'.format(seq),
                        #                             'Aligned camera trajectory: {}'.format(seq), traj_noise_ass, save, save_figure_path, 'noisy')                        
                        visualize_aligned_keyframes(traj_noise_ass_aligned, traj_gt_ass_noise, seq, '',
                                                    'Aligned noisy data: {}'.format(seq), [], save, save_figure_path, 'noisy')
                        visualize_aligned_keyframes(traj_BA_ass_aligned, traj_gt_ass_BA, seq, '',
                                                    'Aligned noisy data after Bundle Adjustment: {}'.format(seq), [], save, save_figure_path, 'BA')
                        visualize_aligned_keyframes(traj_smooth_ass_aligned, traj_gt_ass_smooth, seq, '',
                                                    'Aligned noisy data after Bundle Adjustment (smooth): {}'.format(seq), [], save, save_figure_path, 'smooth')
                        
                    # Calculate APE & RPE values
                    # visualize= 0
                    APE_noise_tran = calculate_APE_simple(traj_gt_ass_noise, traj_noise_ass_aligned, 'translation', 'rmse', visualize, save, save_figure_path, sequence = seq, seq_type='Noisy')
                    APE_BA_tran = calculate_APE_simple(traj_gt_ass_BA, traj_BA_ass_aligned, 'translation', 'rmse', visualize, save, save_figure_path, sequence = seq, seq_type='After BA')
                    APE_smooth_tran = calculate_APE_simple(traj_gt_ass_smooth, traj_smooth_ass_aligned, 'translation', 'rmse', visualize, save, save_figure_path, sequence = seq, seq_type='Smooth')

                    RPE_noise_tran = calculate_RPE_simple(traj_gt_ass_noise, traj_noise_ass_aligned, 'translation', 'rmse', visualize, save, save_figure_path, sequence = seq, seq_type='Noisy')
                    RPE_BA_tran = calculate_RPE_simple(traj_gt_ass_BA, traj_BA_ass_aligned, 'translation', 'rmse',  visualize, save, save_figure_path, sequence = seq, seq_type='After BA')
                    RPE_smooth_tran = calculate_RPE_simple(traj_gt_ass_smooth, traj_smooth_ass_aligned, 'translation', 'rmse', visualize, save, save_figure_path, sequence = seq, seq_type='Smooth')

                    APE_noise_rot = calculate_APE_simple(traj_gt_ass_noise, traj_noise_ass_aligned, 'rotation angle degrees', 'rmse', visualize, save, save_figure_path, sequence = seq, seq_type='Noisy')
                    APE_BA_rot = calculate_APE_simple(traj_gt_ass_BA, traj_BA_ass_aligned, 'rotation angle degrees', 'rmse',  visualize, save, save_figure_path, sequence = seq, seq_type='After BA')
                    APE_smooth_rot = calculate_APE_simple(traj_gt_ass_smooth, traj_smooth_ass_aligned, 'rotation angle degrees', 'rmse', visualize, save, save_figure_path, sequence = seq, seq_type='Smooth')

                    RPE_noise_rot = calculate_RPE_simple(traj_gt_ass_noise, traj_noise_ass_aligned, 'rotation angle degrees', 'rmse',  visualize, save, save_figure_path, sequence = seq, seq_type='Noisy')
                    RPE_BA_rot = calculate_RPE_simple(traj_gt_ass_BA, traj_BA_ass_aligned, 'rotation angle degrees', 'rmse', visualize, save, save_figure_path, sequence = seq, seq_type='After BA')
                    RPE_smooth_rot = calculate_RPE_simple(traj_gt_ass_smooth, traj_smooth_ass_aligned, 'rotation angle degrees', 'rmse', visualize, save, save_figure_path, sequence = seq, seq_type='Smooth')


                    if limit =='Limited': # Append rmse value to calculate statistics
                        APE_rmse_noise_lim_tran.append(APE_noise_tran), APE_rmse_BA_lim_tran.append(APE_BA_tran), APE_rmse_smooth_lim_tran.append(APE_smooth_tran)
                        RPE_rmse_noise_lim_tran.append(RPE_noise_tran), RPE_rmse_BA_lim_tran.append(RPE_BA_tran), RPE_rmse_smooth_lim_tran.append(RPE_smooth_tran)

                        APE_rmse_noise_lim_rot.append(APE_noise_rot), APE_rmse_BA_lim_rot.append(APE_BA_rot), APE_rmse_smooth_lim_rot.append(APE_smooth_rot)
                        RPE_rmse_noise_lim_rot.append(RPE_noise_rot), RPE_rmse_BA_lim_rot.append(RPE_BA_rot), RPE_rmse_smooth_lim_rot.append(RPE_smooth_rot)
                    elif limit == 'No limits':
                        APE_rmse_noise_unlim_tran.append(APE_noise_tran), APE_rmse_BA_unlim_tran.append(APE_BA_tran), APE_rmse_smooth_unlim_tran.append(APE_smooth_tran)
                        RPE_rmse_noise_unlim_tran.append(RPE_noise_tran), RPE_rmse_BA_unlim_tran.append(RPE_BA_tran), RPE_rmse_smooth_unlim_tran.append(RPE_smooth_tran)

                        APE_rmse_noise_unlim_rot.append(APE_noise_rot), APE_rmse_BA_unlim_rot.append(APE_BA_rot), APE_rmse_smooth_unlim_rot.append(APE_smooth_rot)
                        RPE_rmse_noise_unlim_rot.append(RPE_noise_rot), RPE_rmse_BA_unlim_rot.append(RPE_BA_rot), RPE_rmse_smooth_unlim_rot.append(RPE_smooth_rot)

                # iterations.append(iter), cost_reduction.append(cost_red), timings.append(time)

                # Calculate full statistics
                # APE_stats_trans = calculate_APE_simple(traj_gt_ass_noise, traj_noise_ass_aligned, 'translation', 'all', 0, 0, save_fig_path, sequence = seq, seq_type='Noisy')
                # APE_stats_rot = calculate_APE_simple(traj_gt_ass_noise, traj_noise_ass_aligned, 'rotation angle degrees', 'all', 0, 0, save_fig_path, sequence = seq, seq_type='Noisy')

                # RPE_stats_trans = calculate_RPE_simple(traj_gt_ass_noise, traj_noise_ass_aligned, 'translation', 'all', 0, 0, save_fig_path, sequence = seq, seq_type='Noisy')
                # RPE_stats_rot = calculate_RPE_simple(traj_gt_ass_noise, traj_noise_ass_aligned, 'rotation angle degrees', 'all', 0, 0, save_fig_path, sequence = seq, seq_type='Noisy')

                # stats_APE = [APE_stats_trans, APE_stats_rot]
                # stats_RPE = [RPE_stats_trans, RPE_stats_rot]

                # if limit == 'No limits':
                #     saveARPE2csv("APE", stats_APE, sequence)
                #     saveARPE2csv("RPE", stats_RPE, sequence)

            iterations.append(iter), cost_reduction.append(cost_red), timings.append(time) # => should be here...
            # visualize = 1
    # Merge output into 1 data struct
    APE_lim_tran = np.array([APE_rmse_noise_lim_tran, APE_rmse_BA_lim_tran, APE_rmse_smooth_lim_tran])
    RPE_lim_tran = np.array([RPE_rmse_noise_lim_tran, RPE_rmse_BA_lim_tran, RPE_rmse_smooth_lim_tran])

    APE_unlim_tran = np.array([APE_rmse_noise_unlim_tran, APE_rmse_BA_unlim_tran, APE_rmse_smooth_unlim_tran])
    RPE_unlim_tran = np.array([RPE_rmse_noise_unlim_tran, RPE_rmse_BA_unlim_tran, RPE_rmse_smooth_unlim_tran])

    APE_lim_rot = np.array([APE_rmse_noise_lim_rot, APE_rmse_BA_lim_rot, APE_rmse_smooth_lim_rot])
    RPE_lim_rot = np.array([RPE_rmse_noise_lim_rot, RPE_rmse_BA_lim_rot, RPE_rmse_smooth_lim_rot])

    APE_unlim_rot = np.array([APE_rmse_noise_unlim_rot, APE_rmse_BA_unlim_rot, APE_rmse_smooth_unlim_rot])
    RPE_unlim_rot = np.array([RPE_rmse_noise_unlim_rot, RPE_rmse_BA_unlim_rot, RPE_rmse_smooth_unlim_rot])

    return APE_lim_tran, RPE_lim_tran, APE_unlim_tran, RPE_unlim_tran, APE_lim_rot, RPE_lim_rot, APE_unlim_rot, RPE_unlim_rot, iterations, cost_reduction, timings



# Parallelize for sensitivity analysis
def run_BA_noise(args):
    return BA_noise(*args)

def BA_noise_parallel(args):
    nbr_thread = args[0]
    noise_level = args[11]

    # Create a pool of worker processes
    pool = mp.Pool(len(noise_level))

    # Split the task into smaller chunks
    args_chunks = []
    for i in range(len(noise_level)):
        args_copy = copy.deepcopy(args)  # Create a new copy of args
        args_copy[0] = nbr_thread[i] # Modify the copy
        args_copy[11] = [noise_level[i]]  
        args_chunks.append(args_copy)

    # Run the bundle adjustment for each chunck in parallel
    results = pool.map(run_BA_noise, args_chunks)

    # Merge the results from all chunks
    results_cat = []
    for i in range(len(results[0])):
        res = [result[i] for result in results]
        results_cat.append(res)

    # Close the pool of worker processes
    pool.close()

    return results_cat
