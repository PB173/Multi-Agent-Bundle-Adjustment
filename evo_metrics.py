# This file contains functions that will calculate the metrics to compare the estimated trajectories to the provided
# ground truth.

from functions import saveARPE2csv

from evo.core import metrics, sync
from evo.tools import file_interface, plot
import copy
import matplotlib.pyplot as plt
# import Evo.evo_metrics as evo_metrics

# Functions to align trajectories to provided ground truth
def align_BA_estimates(ORB_type, ORB_version, sequence, Th_val, do_plot, print_stats, save):
    gt_path = '/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Ground Truth/state_estimate/TUM/'.format(ORB_version)
    BA_path = '/home/pb/Documents/Thesis/Data/SABA/{}/{}/TUM/{}/'.format(ORB_version, ORB_type, Th_val)

    # Read TUM trajectory files of ground truth and estimations
    traj_gt = file_interface.read_tum_trajectory_file(gt_path + 'gt_{}_TUM.txt'.format(sequence))
    traj_BA = file_interface.read_tum_trajectory_file(BA_path + 'BA_kf_{}_TUM.txt'.format(sequence))
    if print_stats:
        print("Ground truth before sync:", traj_gt)
        print("Estimations before sync;", traj_BA)

    # Synchronize estimations and ground truth based on timestamp
    max_diff = 0.01
    traj_gt, traj_BA = sync.associate_trajectories(traj_gt, traj_BA, max_diff)
    if print_stats:
        print("\nGround truth after sync:", traj_gt)
        print("Estimations after sync;", traj_BA)

    # Align estimations and ground truth and correct scale (for monocular images)
    traj_BA_aligned = copy.deepcopy(traj_BA)
    traj_BA_aligned.align(traj_gt, correct_scale=True, correct_only_scale=False)
    if print_stats:
        print("\nEstimated trajectory after alignment:", traj_BA_aligned)

    if do_plot:
        plot_mode = plot.PlotMode.xyz

        fig = plt.figure()
        ax = plot.prepare_axis(fig, plot_mode)
        fig.set_size_inches(32, 18)

        plot.traj(ax, plot_mode, traj_BA, '-', "blue", "Camera trajectory after BA (unaligned)")
        plot.traj(ax, plot_mode, traj_BA_aligned, '-', "green", "Camera trajectory after BA (aligned)")
        plot.traj(ax, plot_mode, traj_gt, '-', "red", "Ground truth")
                
        # set axis limits for 3D plot depending on sequence
        if sequence == "MH01":
            ax.set_xlim([-2.5, 4]), ax.set_ylim([-3.5, 3]), ax.set_zlim([-2.5, 4])
        elif sequence == "MH02":
            ax.set_xlim([-5, 20]), ax.set_ylim([-10, 15]), ax.set_zlim([-2, 18])
        elif sequence == "MH03":
            ax.set_xlim([-4, 8.5]), ax.set_ylim([-6, 6]), ax.set_zlim([-6, 6])
        elif sequence == "MH04":
            ax.set_xlim([-3, 13]), ax.set_ylim([-4, 8]), ax.set_zlim([-8, 7])
        elif sequence == "MH05":
            ax.set_xlim([-20, 25]), ax.set_ylim([-15, 15]), ax.set_zlim([-2, 15])

        ax.set_title("Alignment of the camera trajectory after BA with the ground truth: " + sequence, pad=70, fontsize = 30,
                    fontweight = "bold")
        if save:
            plt.savefig("/home/pb/Documents/Thesis/Figures/ORB SLAM/Evo/Estimates_Ground/est_gt_" + sequence + ".svg", 
                bbox_inches = 'tight', pad_inches = 0, transparent = False)
        
        fig = plt.figure()
        ax = plot.prepare_axis(fig, plot_mode)
        fig.set_size_inches(32, 18)

        plot.traj(ax, plot_mode, traj_BA_aligned, '-', "blue", "Camera trajectory after BA (aligned)")
        plot.traj(ax, plot_mode, traj_gt, '-', "green", "Ground truth")
                
        # set axis limits for 3D plot depending on sequence
        if sequence == "MH01":
            ax.set_xlim([-1.5, 3.5]), ax.set_ylim([1, 6]), ax.set_zlim([-2.5, 2.5])
        elif sequence == "MH02":
            ax.set_xlim([-1, 3.5]), ax.set_ylim([1.5, 6]), ax.set_zlim([-2, 2.5])
        elif sequence == "MH03":
            ax.set_xlim([1.5, 9.5]), ax.set_ylim([-1.5, 6]), ax.set_zlim([-2.5, 4])
        elif sequence == "MH04":
            ax.set_xlim([1.5, 14.5]), ax.set_ylim([-3, 10]), ax.set_zlim([-4.5, 7])
        elif sequence == "MH05":
            ax.set_xlim([1.5, 14.5]), ax.set_ylim([-2.5, 9]), ax.set_zlim([-2, 6])

        ax.set_title("Aligned estimates after BA vs ground truth: " + sequence, pad=70, fontsize = 30,
                    fontweight = "bold")
        if save:
            plt.savefig("/home/pb/Documents/Thesis/Figures/ORB SLAM/Evo/Estimates_Ground/aligned_est_gt_" + sequence + ".svg", 
            bbox_inches = 'tight', pad_inches = 0, transparent = False)
        
        plt.show()

    return traj_gt, traj_BA, traj_BA_aligned

def align_keyframes(ORB_type, ORB_version, sequence, do_plot, title_1, title_2, print_stats, save):
    gt_path = '/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Ground Truth/state_estimate/TUM/'.format(ORB_version)
    kf_path = '/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Keyframes/{}/TUM/'.format(ORB_version, ORB_type)

    # Read TUM trajectory files of ground truth and estimations
    traj_gt = file_interface.read_tum_trajectory_file(gt_path + 'gt_{}_TUM.txt'.format(sequence))
    traj_kf = file_interface.read_tum_trajectory_file(kf_path + '{}_kf_{}_TUM.txt'.format(ORB_version, sequence))
    if print_stats:
        print("\nGround truth before sync:", traj_gt)
        print("Keyframes before sync:", traj_kf)

    # Synchronize estimations and ground truth based on timestamp
    max_diff = 0.01
    traj_gt, traj_kf = sync.associate_trajectories(traj_gt, traj_kf, max_diff)
    if print_stats:
        print("Ground truth after sync:", traj_gt)
        print("Keyframes after sync;", traj_kf)

    # Align keyframes and ground truth and correct scale (for monocular images)
    traj_kf_aligned = copy.deepcopy(traj_kf)
    traj_kf_aligned.align(traj_gt, correct_scale=True, correct_only_scale=False)
    if print_stats:
        print("Camera trajectory after alignment:", traj_kf_aligned)

    if do_plot:
        visualize_aligned_keyframes(traj_kf, traj_kf_aligned, traj_gt, sequence, title_1, title_2, save)

    return traj_gt, traj_kf, traj_kf_aligned

def visualize_aligned_keyframes(traj_kf_aligned, traj_gt, sequence, title_1, title_2, traj_kf=[], save=0, save_fig_path='', type_traj=''):
    plot_mode = plot.PlotMode.xyz

    if traj_kf != []:
        fig = plt.figure()
        ax = plot.prepare_axis(fig, plot_mode)
        fig.set_size_inches(32, 18)
        plot.traj(ax, plot_mode, traj_kf, '-', "blue", "Camera trajectory (unaligned)")
        plot.traj(ax, plot_mode, traj_kf_aligned, '-', "green", "Camera trajectory (aligned)")
        plot.traj(ax, plot_mode, traj_gt, '-', "red", "Ground truth")
                
        # set axis limits for 3D plot depending on sequence
        if sequence == "MH01":
            ax.set_xlim([-2.5, 4]), ax.set_ylim([-3.5, 3]), ax.set_zlim([-2.5, 4])
        elif sequence == "MH02":
            ax.set_xlim([-5, 20]), ax.set_ylim([-10, 15]), ax.set_zlim([-2, 18])
        elif sequence == "MH03":
            ax.set_xlim([-4, 8.5]), ax.set_ylim([-6, 6]), ax.set_zlim([-6, 6])
        elif sequence == "MH04":
            ax.set_xlim([-3, 13]), ax.set_ylim([-4, 8]), ax.set_zlim([-8, 7])
        elif sequence == "MH05":
            ax.set_xlim([-20, 25]), ax.set_ylim([-15, 15]), ax.set_zlim([-2, 15])

        ax.set_title(title_1, pad=70, fontsize = 30,
                        fontweight = "bold")
        if save:
            plt.savefig(save_fig_path + 'camera_trajectories_{}'.format(type_traj), bbox_inches = 'tight', pad_inches = 0, transparent = False)
            plt.close('all')

    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    fig.set_size_inches(32, 18)

    plot.traj(ax, plot_mode, traj_kf_aligned, '-', "blue", "Camera trajectory (aligned)")
    plot.traj(ax, plot_mode, traj_gt, '-', "green", "Ground truth")
            
    # set axis limits for 3D plot depending on sequence
    if sequence == "MH01":
        ax.set_xlim([-1.5, 3.5]), ax.set_ylim([1, 6]), ax.set_zlim([-2.5, 2.5])
    elif sequence == "MH02":
        ax.set_xlim([-1, 3.5]), ax.set_ylim([1.5, 6]), ax.set_zlim([-2, 2.5])
    elif sequence == "MH03":
        ax.set_xlim([1.5, 9.5]), ax.set_ylim([-1.5, 6]), ax.set_zlim([-2.5, 4])
    elif sequence == "MH04":
        ax.set_xlim([1.5, 14.5]), ax.set_ylim([-3, 10]), ax.set_zlim([-4.5, 7])
    elif sequence == "MH05":
        ax.set_xlim([1.5, 14.5]), ax.set_ylim([-2.5, 9]), ax.set_zlim([-2, 6])

    ax.set_title(title_2, pad=90, fontsize = 40,
                    fontweight = "bold")
    if save:
        plt.savefig(save_fig_path + 'aligned_camera_trajectory_{}.png'.format(type_traj), bbox_inches = 'tight', pad_inches = 0.5, transparent = False)
        plt.close('all')
    else:
        plt.show()



def visualize_results_aligned(trajectories, save=0, save_fig_path = ''):

    
    x_idx, y_idx, z_idx = plot.plot_mode_to_idx(plot.PlotMode.xyz)
    scale = [150, 50, 50, 5, 1]
    markers = ["x", "o", "^", "s", "s"]
    colors = ["black", "red", "green", "cyan", "yellow"]
    
    # scale = [200, 150, 50, 50, 2]
    # markers = ["s", "x", "o", "^", "s"]
    # colors = ["yellow", "black", "red", "green", "cyan"]

    # Initialize figures
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(32, 18)
    fig.subplots_adjust(top=0.96, bottom=0.06, left=0.06, right=0.99, hspace=0.45, wspace=0.2)

    for i, traj in enumerate(trajectories):
        x = traj.positions_xyz[:, x_idx]
        y = traj.positions_xyz[:, y_idx]
        z = traj.positions_xyz[:, z_idx]

        # print("{} : \n".format(i), x)


        ax[0].scatter(range(len(x)), x, linewidths = 3, marker = markers[i], color = colors[i], s = scale[i])
        ax[1].scatter(range(len(y)), y, linewidths = 3, marker = markers[i], color = colors[i], s = scale[i])
        ax[2].scatter(range(len(z)), z, linewidths = 3, marker = markers[i], color = colors[i], s = scale[i])

    labels= ('Original camera position', 'Noisy camera position', 'After BA', 'After BA (smooth)', 'Ground Truth')
    ax[0].set_title('X position', fontsize = 40, fontweight ='bold')
    ax[1].set_title('Y position', fontsize = 40, fontweight ='bold')
    ax[2].set_title('Z position', fontsize = 40, fontweight ='bold') 

    for axis in ax:
        axis.set_xlabel('Keyframe', fontsize = 30), axis.set_ylabel('Position', fontsize = 30)
        axis.legend(labels, fontsize = 24, markerscale=2)
        axis.minorticks_on(), axis.ticklabel_format(useOffset=False)
        axis.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
        axis.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')
        axis.set_xlim([0, len(x)+75])

        axis.tick_params(axis='both', which='major', labelsize=24)
    
    if save:
        plt.savefig(save_fig_path + 'camera_poses_2D_ground_truth.png', bbox_inches = 'tight', pad_inches = 0.5, transparent = False)
        plt.close('all')
    else:
        plt.show()





# Functions to calculate certain metrics
def calculate_APE(type_traj, traj_gt, traj_BA, mode, stats, do_plot, save, sequence=""):
    if mode == "translation":
        pose_relation = metrics.PoseRelation.translation_part
    elif mode == "rotation":
        pose_relation = metrics.PoseRelation.rotation_part
    elif mode == "rotation angle degrees":
        pose_relation = metrics.PoseRelation.rotation_angle_deg
    elif mode == "rotation angle radians":
        pose_relation = metrics.PoseRelation.rotation_angle_rad
    elif mode == "full":
        pose_relation = metrics.PoseRelation.full_transformation
    else:
        raise ValueError("Invalid input. Please choose between 'translation, 'rotation'," 
                         "'rotation angle degrees/radians' or 'full'. Exiting now.")

    data = (traj_gt, traj_BA)

    # Run APE on data
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)

    # ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)

    ape_stats = ape_metric.get_all_statistics()

    if "all" in stats:
        ape_stat = ape_stats
    else:
        ape_stat = ape_stats[str(stats)]
        
    # Optionally, plot the data 
    if do_plot:
        # Plot all statisctics on 2D plot
        seconds_from_start = [t - traj_BA.timestamps[0] for t in traj_BA.timestamps]
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        plot.error_array(fig.gca(), ape_metric.error, x_array=seconds_from_start,
                        statistics={s:v for s,v in ape_stats.items() if s != "sse"},
                        name="APE", title="APE statistics w.r.t. " + ape_metric.pose_relation.value +": " + sequence, xlabel="$t$ (s)")
        if save:
            plt.savefig("/home/pb/Documents/Thesis/Figures/ORB SLAM/Evo/APE/{}/Statistics/APE_stats_".format(type_traj) + sequence + "_" + mode +".svg", 
                        bbox_inches = 'tight', pad_inches = 0.2)
            
        # Plot difference between trajectories (2D)
        plot_mode = plot.PlotMode.xy
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        ax = plot.prepare_axis(fig, plot_mode)
        plot.traj(ax, plot_mode, traj_gt, '--', "gray", "Ground truth")
        plot.traj_colormap(ax, traj_BA, ape_metric.error, 
                        plot_mode, min_map=ape_stats["min"], max_map=ape_stats["max"])
        ax.legend(fontsize=20)
        ax.set_title("APE w.r.t. " + ape_metric.pose_relation.value +", 2D: " + sequence, fontsize = 24, fontweight = "bold")
        if save:
            plt.savefig("/home/pb/Documents/Thesis/Figures/ORB SLAM/Evo/APE/{}/2D trajectories/APE_2D_".format(type_traj) + sequence + "_" + 
                        mode +".svg", 
                        transparent = False, bbox_inches = 'tight', pad_inches = 0.2)
        
        # Plot difference between trajectories (3D)
        plot_mode = plot.PlotMode.xyz
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        ax = plot.prepare_axis(fig, plot_mode)
        plot.traj(ax, plot_mode, traj_gt, '--', "gray", "Ground truth")
        plot.traj_colormap(ax, traj_BA, ape_metric.error, 
                        plot_mode, min_map=ape_stats["min"], max_map=ape_stats["max"])
        ax.legend(fontsize=20)
        ax.set_title("APE w.r.t. " + ape_metric.pose_relation.value +", 3D: " + sequence, pad=50, fontsize = 24,
                     fontweight = "bold")

        # set axis limits for 3D plot depending on sequence
        if sequence == "MH01":
            ax.set_xlim([-1.5, 3.5]), ax.set_ylim([1, 6]), ax.set_zlim([-2.5, 2.5])
        elif sequence == "MH02":
            ax.set_xlim([-1, 3.5]), ax.set_ylim([1.5, 6]), ax.set_zlim([-2, 2.5])
        elif sequence == "MH03":
            ax.set_xlim([1.5, 9.5]), ax.set_ylim([-1.5, 6]), ax.set_zlim([-2.5, 4])
        elif sequence == "MH04":
            ax.set_xlim([1.5, 14.5]), ax.set_ylim([-3, 10]), ax.set_zlim([-4.5, 7])
        elif sequence == "MH05":
            ax.set_xlim([1.5, 14.5]), ax.set_ylim([-2.5, 9]), ax.set_zlim([-2, 6])

        if save:
            plt.savefig("/home/pb/Documents/Thesis/Figures/ORB SLAM/Evo/APE/{}/3D trajectories/APE_3D_".format(type_traj) + sequence + "_" + mode +".svg", 
                        transparent = False, bbox_inches = 'tight', pad_inches = 0.2)
        # plt.show()

    return ape_stat

def calculate_RPE(type_traj, traj_gt, traj_BA, mode, stats, do_plot, save, sequence=""):
    if mode == "translation":
        pose_relation = metrics.PoseRelation.translation_part
    elif mode == "rotation":
        pose_relation = metrics.PoseRelation.rotation_part
    elif mode == "rotation angle degrees":
        pose_relation = metrics.PoseRelation.rotation_angle_deg
    elif mode == "rotation angle radians":
        pose_relation = metrics.PoseRelation.rotation_angle_rad
    elif mode == "full":
        pose_relation = metrics.PoseRelation.full_transformation
    else:
        raise ValueError("Invalid input. Please choose between 'translation, 'rotation'," 
                         "'rotation angle degrees/radians' or 'full'. Exiting now.")
    
    # normal mode
    delta = 0.5
    delta_unit = metrics.Unit.meters
    
    # all pairs mode
    all_pairs = False  # activate

    data = (traj_gt, traj_BA)

    # Run RPE on data
    rpe_metric = metrics.RPE(pose_relation=pose_relation, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
    rpe_metric.process_data(data)

    rpe_stats = rpe_metric.get_all_statistics()

    if "all" in stats:
        rpe_stat = rpe_stats
    else:
        rpe_stat = rpe_stats[str(stats)]
    
    # Optionally, plot the data 
    if do_plot:
        traj_gt_plot = copy.deepcopy(traj_gt)
        traj_BA_plot = copy.deepcopy(traj_BA)
        traj_gt_plot.reduce_to_ids(rpe_metric.delta_ids)
        traj_BA_plot.reduce_to_ids(rpe_metric.delta_ids)

        seconds_from_start = [t - traj_BA_plot.timestamps[0] for t in traj_BA_plot.timestamps]

        # Plot all statistics on a 2D plot
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        plot.error_array(fig.gca(), rpe_metric.error, x_array=seconds_from_start,
                        statistics={s:v for s,v in rpe_stats.items() if s != "sse"},
                        name="RPE", title="RPE statistics w.r.t. " + rpe_metric.pose_relation.value +": " + sequence, xlabel="$t$ (s)")
        if save:
            plt.savefig("/home/pb/Documents/Thesis/Figures/ORB SLAM/Evo/RPE/{}/Statistics/RPE_stats_".format(type_traj) + sequence + "_" + mode +".svg", 
                        bbox_inches = 'tight', pad_inches = 0.2)
            
        # Plot difference between trajectories (2D)
        plot_mode = plot.PlotMode.xy
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        ax = plot.prepare_axis(fig, plot_mode)
        plot.traj(ax, plot_mode, traj_gt_plot, '--', "gray", "Ground truth")
        plot.traj_colormap(ax, traj_BA_plot, rpe_metric.error, plot_mode, min_map=rpe_stats["min"], max_map=rpe_stats["max"])
        ax.legend(fontsize=20)
        ax.set_title("RPE w.r.t. " + rpe_metric.pose_relation.value +", 2D: " + sequence, fontsize = 22, fontweight = "bold")
        if save:
            plt.savefig("/home/pb/Documents/Thesis/Figures/ORB SLAM/Evo/RPE/{}/2D trajectories/RPE_2D_".format(type_traj) + sequence + "_" + 
                        mode +".svg", 
                        transparent = False, bbox_inches = 'tight', pad_inches = 0.2)

        # Plot difference between trajectories (3D)
        plot_mode = plot.PlotMode.xyz
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        ax = plot.prepare_axis(fig, plot_mode)
        plot.traj(ax, plot_mode, traj_gt_plot, '--', "gray", "Ground truth")
        plot.traj_colormap(ax, traj_BA_plot, rpe_metric.error, plot_mode, min_map=rpe_stats["min"], max_map=rpe_stats["max"])
        ax.legend(fontsize=20)
        ax.set_title("RPE w.r.t. " + rpe_metric.pose_relation.value +", 3D: " + sequence, pad=50, fontsize = 24,
                     fontweight = "bold")

        # set axis limits for 3D plot depending on sequence
        if sequence == "MH01":
            ax.set_xlim([-1.5, 3.5]), ax.set_ylim([1, 6]), ax.set_zlim([-2.5, 2.5])
        elif sequence == "MH02":
            ax.set_xlim([-1, 3.5]), ax.set_ylim([1.5, 6]), ax.set_zlim([-2, 2.5])
        elif sequence == "MH03":
            ax.set_xlim([1.5, 9.5]), ax.set_ylim([-1.5, 6]), ax.set_zlim([-2.5, 4])
        elif sequence == "MH04":
            ax.set_xlim([1.5, 14.5]), ax.set_ylim([-3, 10]), ax.set_zlim([-4.5, 7])
        elif sequence == "MH05":
            ax.set_xlim([1.5, 14.5]), ax.set_ylim([-2.5, 9]), ax.set_zlim([-2, 6])
        
        if save:
            plt.savefig("/home/pb//Documents/Thesis/Figures/ORB SLAM/Evo/RPE/{}/3D trajectories/RPE_3D_".format(type_traj) + sequence + "_" + mode +".svg", 
                        transparent = False, bbox_inches = 'tight', pad_inches = 0.2)

        # plt.show()
    
    return rpe_stat

def calculate_APE_simple(traj_gt, traj_est, mode, stats, visualize = 0, save = 0, save_fig_path = '', sequence = '', seq_type = ''):
    if mode == "translation":
        pose_relation = metrics.PoseRelation.translation_part
    elif mode == "rotation":
        pose_relation = metrics.PoseRelation.rotation_part
    elif mode == "rotation angle degrees":
        pose_relation = metrics.PoseRelation.rotation_angle_deg
    elif mode == "rotation angle radians":
        pose_relation = metrics.PoseRelation.rotation_angle_rad
    elif mode == "full":
        pose_relation = metrics.PoseRelation.full_transformation
    else:
        raise ValueError("Invalid input. Please choose between 'translation, 'rotation'," 
                         "'rotation angle degrees/radians' or 'full'. Exiting now.")

    data = (traj_gt, traj_est)

    # Run APE on data
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)

    # ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)

    ape_stats = ape_metric.get_all_statistics()

    if "all" in stats:
        ape_stat = ape_stats
    else:
        ape_stat = ape_stats[str(stats)]

    # Optionally, plot the data 
    if visualize:
        # Plot all statisctics on 2D plot
        seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est.timestamps]
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        plot.error_array(fig.gca(), ape_metric.error, x_array=seconds_from_start,
                        statistics={s:v for s,v in ape_stats.items() if s != "sse"},
                        name="APE", title="APE statistics w.r.t. " + ape_metric.pose_relation.value +": " + sequence + "- " + seq_type, xlabel="$t$ (s)")
        if save:
            plt.savefig(save_fig_path + 'APE_{}_stats_{}.png'.format(pose_relation.value, seq_type), bbox_inches = 'tight', pad_inches = 0, transparent = False)
            plt.close('all')
        # save_fig_path = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Illustrations/Alignment/'
        # plt.savefig(save_fig_path + 'APE_{}_stats_{}_{}.png'.format(pose_relation.value, seq_type, sequence), bbox_inches = 'tight', pad_inches = 0.5, transparent = False)

            
        # Plot difference between trajectories (2D)
        plot_mode = plot.PlotMode.xy
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        ax = plot.prepare_axis(fig, plot_mode)
        plot.traj(ax, plot_mode, traj_gt, '--', "gray", "Ground Truth")
        plot.traj_colormap(ax, traj_est, ape_metric.error, 
                        plot_mode, min_map=ape_stats["min"], max_map=ape_stats["max"])
        ax.legend(fontsize=20)
        ax.set_title("APE w.r.t. " + ape_metric.pose_relation.value +", 2D: " + sequence +"- " + seq_type, fontsize = 24, fontweight = "bold")
        
        if save:
            plt.savefig(save_fig_path + 'APE_{}_2D_{}.png'.format(pose_relation.value, seq_type), bbox_inches = 'tight', pad_inches = 0, transparent = False)
            plt.close('all')      
       
        # Plot difference between trajectories (3D)
        plot_mode = plot.PlotMode.xyz
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        ax = plot.prepare_axis(fig, plot_mode)
        plot.traj(ax, plot_mode, traj_gt, '--', "gray", "Ground truth")
        plot.traj_colormap(ax, traj_est, ape_metric.error, 
                        plot_mode, min_map=ape_stats["min"], max_map=ape_stats["max"])
        ax.legend(fontsize=20)
        ax.set_title("APE w.r.t. " + ape_metric.pose_relation.value +", 3D: " + sequence +"- " + seq_type, pad=50, fontsize = 24, fontweight = "bold")


        # set axis limits for 3D plot depending on sequence
        if sequence == "MH01":
            ax.set_xlim([-1.5, 3.5]), ax.set_ylim([1, 6]), ax.set_zlim([-2.5, 2.5])
        elif sequence == "MH02":
            ax.set_xlim([-1, 3.5]), ax.set_ylim([1.5, 6]), ax.set_zlim([-2, 2.5])
        elif sequence == "MH03":
            ax.set_xlim([1.5, 9.5]), ax.set_ylim([-1.5, 6]), ax.set_zlim([-2.5, 4])
        elif sequence == "MH04":
            ax.set_xlim([1.5, 14.5]), ax.set_ylim([-3, 10]), ax.set_zlim([-4.5, 7])
        elif sequence == "MH05":
            ax.set_xlim([1.5, 14.5]), ax.set_ylim([-2.5, 9]), ax.set_zlim([-2, 6])
        
        if save:
            plt.savefig(save_fig_path + 'APE_{}_3D_{}.png'.format(pose_relation.value, seq_type), bbox_inches = 'tight', pad_inches = 0, transparent = False)
            plt.close('all')

        
        if not save:
            plt.show()
    
    return ape_stat

def calculate_RPE_simple(traj_gt, traj_est, mode, stats, visualize = 0, save = 0, save_fig_path = '', sequence = '', seq_type = ''):
    if mode == "translation":
        pose_relation = metrics.PoseRelation.translation_part
    elif mode == "rotation":
        pose_relation = metrics.PoseRelation.rotation_part
    elif mode == "rotation angle degrees":
        pose_relation = metrics.PoseRelation.rotation_angle_deg
    elif mode == "rotation angle radians":
        pose_relation = metrics.PoseRelation.rotation_angle_rad
    elif mode == "full":
        pose_relation = metrics.PoseRelation.full_transformation
    else:
        raise ValueError("Invalid input. Please choose between 'translation, 'rotation'," 
                         "'rotation angle degrees/radians' or 'full'. Exiting now.")
    
    # normal mode
    delta = 1
    delta_unit = metrics.Unit.frames
    
    # all pairs mode
    all_pairs = False  # activate

    data = (traj_gt, traj_est)

    # Run RPE on data
    rpe_metric = metrics.RPE(pose_relation=pose_relation, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
    rpe_metric.process_data(data)

    rpe_stats = rpe_metric.get_all_statistics()

    if "all" in stats:
        rpe_stat = rpe_stats
    else:
        rpe_stat = rpe_stats[str(stats)]
    
    # Optionally, plot the data 
    if visualize:
        # Important: restrict data to delta ids for plot
        traj_gt_plot = copy.deepcopy(traj_gt)
        traj_BA_plot = copy.deepcopy(traj_est)
        traj_gt_plot.reduce_to_ids(rpe_metric.delta_ids)
        traj_BA_plot.reduce_to_ids(rpe_metric.delta_ids)

        # Plot all statisctics on 2D plot
        seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est.timestamps[1:]]
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        plot.error_array(fig.gca(), rpe_metric.error, x_array=seconds_from_start,
                        statistics={s:v for s,v in rpe_stats.items() if s != "sse"},
                        name="RPE", title="RPE statistics w.r.t. " + rpe_metric.pose_relation.value +": " + sequence+"- " + seq_type, xlabel="$t$ (s)")
        if save:
            plt.savefig(save_fig_path + 'RPE_{}_stats_{}.png'.format(pose_relation.value, seq_type), bbox_inches = 'tight', pad_inches = 0, transparent = False)
            plt.close('all')
        # save_fig_path = '/home/pb/Documents/Thesis/Figures/ORB SLAM/Illustrations/Alignment/'
        # plt.savefig(save_fig_path + 'RPE_{}_stats_{}_{}.png'.format(pose_relation.value, seq_type, sequence), bbox_inches = 'tight', pad_inches = 0.5, transparent = False)
    
        # Plot difference between trajectories (2D)
        plot_mode = plot.PlotMode.xy
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        ax = plot.prepare_axis(fig, plot_mode)
        plot.traj(ax, plot_mode, traj_gt_plot, '--', "gray", "Ground Truth")
        plot.traj_colormap(ax, traj_BA_plot, rpe_metric.error, plot_mode, min_map=rpe_stats["min"], max_map=rpe_stats["max"])
        ax.legend(fontsize=20)
        ax.set_title("RPE w.r.t. " + rpe_metric.pose_relation.value +", 2D: " + sequence+"- " + seq_type, fontsize = 24, fontweight = "bold")
        if save:
            plt.savefig(save_fig_path + 'RPE_{}_2D_{}.png'.format(pose_relation.value, seq_type), bbox_inches = 'tight', pad_inches = 0, transparent = False)
            plt.close('all')  
        
        # Plot difference between trajectories (3D)
        plot_mode = plot.PlotMode.xyz
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        ax = plot.prepare_axis(fig, plot_mode)
        plot.traj(ax, plot_mode, traj_gt_plot, '--', "gray", "Ground truth")
        plot.traj_colormap(ax, traj_BA_plot, rpe_metric.error, 
                        plot_mode, min_map=rpe_stats["min"], max_map=rpe_stats["max"])
        ax.legend(fontsize=20)
        ax.set_title("RPE w.r.t. " + rpe_metric.pose_relation.value +", 3D: " + sequence+"- " + seq_type, pad=50, fontsize = 24,
                     fontweight = "bold")

        # set axis limits for 3D plot depending on sequence
        if sequence == "MH01":
            ax.set_xlim([-1.5, 3.5]), ax.set_ylim([1, 6]), ax.set_zlim([-2.5, 2.5])
        elif sequence == "MH02":
            ax.set_xlim([-1, 3.5]), ax.set_ylim([1.5, 6]), ax.set_zlim([-2, 2.5])
        elif sequence == "MH03":
            ax.set_xlim([1.5, 9.5]), ax.set_ylim([-1.5, 6]), ax.set_zlim([-2.5, 4])
        elif sequence == "MH04":
            ax.set_xlim([1.5, 14.5]), ax.set_ylim([-3, 10]), ax.set_zlim([-4.5, 7])
        elif sequence == "MH05":
            ax.set_xlim([1.5, 14.5]), ax.set_ylim([-2.5, 9]), ax.set_zlim([-2, 6])
        
        if save:
            plt.savefig(save_fig_path + 'RPE_{}_3D_{}.png'.format(pose_relation.value, seq_type), bbox_inches = 'tight', pad_inches = 0, transparent = False)
            plt.close('all')  
        
        if not save:
            plt.show()
    
    return rpe_stat

# Display (aligned) keyframes / estimates versus ground truth
def visual_compare_gt(type_traj, traj_gt, traj_kf, traj_kf_aligned, sequence, save):
    x_idx, y_idx, z_idx = plot.plot_mode_to_idx(plot.PlotMode.xyz)
    
    x_gt = traj_gt.positions_xyz[:, x_idx]
    y_gt = traj_gt.positions_xyz[:, y_idx]
    z_gt = traj_gt.positions_xyz[:, z_idx]

    x_kf = traj_kf.positions_xyz[:, x_idx]
    y_kf = traj_kf.positions_xyz[:, y_idx]
    z_kf = traj_kf.positions_xyz[:, z_idx]

    x_kf_a = traj_kf_aligned.positions_xyz[:, x_idx]
    y_kf_a = traj_kf_aligned.positions_xyz[:, y_idx]
    z_kf_a = traj_kf_aligned.positions_xyz[:, z_idx]

    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(32, 18)
    
    if sequence == "MH05":
        # fig.subplots_adjust(top=0.973, bottom=0.051, left=0.035, right=0.981, hspace=0.216, wspace=0.2)
        fig.subplots_adjust(top=0.942, bottom=0.089, left=0.064, right=0.989, hspace=0.634, wspace=0.2)
        fig.suptitle('Alignment of the camera trajectory {} w.r.t. the ground truth: {}'.format(type_traj, sequence), fontsize = 30, fontweight = "bold", y =1.)
    else:
        fig.subplots_adjust(top=0.942, bottom=0.089, left=0.064, right=0.989, hspace=0.634, wspace=0.2)
        fig.suptitle('Alignment of the camera trajectory {} w.r.t. the ground truth: {}'.format(type_traj, sequence), fontsize = 30, fontweight = "bold", y =1)

    ax[0].scatter(range(len(x_kf)), x_kf, linewidths = 3, marker = "o",s = 14, label = '{} (unaligned)'.format(type_traj))
    ax[0].scatter(range(len(x_kf_a)), x_kf_a, linewidths = 3, marker = "o",s = 14, label = '{} (aligned)'.format(type_traj))
    ax[0].scatter(range(len(x_gt)), x_gt, linewidths = 3, marker = "_", s = 14, c='r',label = 'Ground Truth')

    ax[1].scatter(range(len(y_kf)), y_kf, linewidths = 3, marker = "o",s = 14, label = '{} (unaligned)'.format(type_traj))
    ax[1].scatter(range(len(y_kf_a)), y_kf_a, linewidths = 3, marker = "o",s = 14, label = '{} (aligned)'.format(type_traj))
    ax[1].scatter(range(len(y_gt)), y_gt, linewidths = 3, marker = "_", s = 14, c='r',label = 'Ground Truth')

    ax[2].scatter(range(len(z_kf)), z_kf, linewidths = 3, marker = "o",s = 14, label = '{} (unaligned)'.format(type_traj))
    ax[2].scatter(range(len(z_kf_a)), z_kf_a, linewidths = 3, marker = "o",s = 14, label = '{} (aligned)'.format(type_traj))
    ax[2].scatter(range(len(z_gt)), z_gt, linewidths = 3, marker = "_", s = 14, c='r',label = 'Ground Truth')
    

    y_labels = ["x Position", "y Position", "z Position"]

    for i, axis in enumerate(ax):
        axis.set_xlabel('Keyframe', fontsize = 24), axis.set_ylabel(y_labels[i], fontsize = 24)
        if i == 0:
            axis.legend(fontsize = 24, markerscale=2)
            axis.set_xlim([0, len(x_kf)+80])
        else:
            axis.set_xlim([0, len(x_kf)])
        axis.minorticks_on(), axis.ticklabel_format(useOffset=False)
        axis.tick_params(axis='both', which='major', labelsize=24)
        axis.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
        axis.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')
        
    if save:
        plt.savefig("/home/pb/Documents/Thesis/Figures/ORB SLAM/Evo/Keyframes_Ground/Scatter/scattered_kf_gt_" + sequence +".svg", 
                        bbox_inches = 'tight', pad_inches = 0.2, transparent = True)
    plt.show()

# Calculate metrics
def calculate_metrics(type_traj, ORB_type, ORB_version, sequences, Th_val, stat, save, figures=0):
    for sequence in sequences:
        if type_traj == "before":
            traj_gt, traj_kf, traj_kf_aligned = align_keyframes(ORB_type, ORB_version, sequence, do_plot = figures, title_1='', title_2='', print_stats = 0, save=save)
        elif type_traj == "after":
            traj_gt, traj_kf, traj_kf_aligned = align_BA_estimates(ORB_type, ORB_version, sequence, Th_val, do_plot = figures, title_1='', title_2='', print_stats = 0, save=save)
        
        if figures:
           visual_compare_gt(type_traj, traj_gt, traj_kf, traj_kf_aligned, sequence, save=save)
        
        if stat == "both":
            # Calculate metrics (APE, RPE) and statistics
            stats_APE_trans = calculate_APE(type_traj, traj_gt, traj_kf_aligned, "translation", "all", figures, save, sequence)
            stats_RPE_trans = calculate_RPE(type_traj, traj_gt, traj_kf_aligned, "translation", "all", figures, save, sequence)

            stats_APE_rot = calculate_APE(type_traj, traj_gt, traj_kf_aligned, "rotation", "all", figures, save, sequence)
            stats_RPE_rot = calculate_RPE(type_traj, traj_gt, traj_kf_aligned, "rotation", "all", figures, save, sequence)

            stats_APE_full = calculate_APE(type_traj, traj_gt, traj_kf_aligned, "full", "all", figures, save, sequence)
            stats_RPE_full = calculate_RPE(type_traj, traj_gt, traj_kf_aligned, "full", "all", figures, save, sequence)

            stats_APE = [stats_APE_trans, stats_APE_rot, stats_APE_full]
            stats_RPE = [stats_RPE_trans, stats_RPE_rot, stats_RPE_full]
            
            if save:
                saveARPE2csv(type_traj, "APE", stats_APE, sequence)
                saveARPE2csv(type_traj, "RPE", stats_RPE, sequence)
            return stats_APE, stats_RPE

        elif stat == "APE":
             # Calculate metrics (APE) and statistics
            stats_APE_trans = calculate_APE(type_traj, traj_gt, traj_kf_aligned, "translation", "all", figures, save, sequence)
            stats_APE_trans_unaligned = calculate_APE(type_traj, traj_gt, traj_kf, "translation", "all", figures, save, sequence)
            
            stats_APE_rot = calculate_APE(type_traj, traj_gt, traj_kf_aligned, "rotation", "all", figures, save, sequence)
            
            stats_APE_full = calculate_APE(type_traj, traj_gt, traj_kf_aligned, "full", "all", figures, save, sequence)
            
            stats_APE = [stats_APE_trans, stats_APE_trans_unaligned, stats_APE_rot, stats_APE_full]

            if save:
                saveARPE2csv(type_traj, "APE", stats_APE, sequence)
            return stats_APE

        elif stat == "RPE":
             # Calculate metrics (RPE) and statistics
            stats_RPE_trans = calculate_RPE(type_traj, traj_gt, traj_kf_aligned, "translation", "all", figures, save, sequence)
            stats_RPE_rot = calculate_RPE(type_traj, traj_gt, traj_kf_aligned, "rotation", "all", figures, save, sequence)
            
            stats_RPE_full = calculate_RPE(type_traj, traj_gt, traj_kf_aligned, "full", "all", figures, sequence)

            stats_RPE = [stats_RPE_trans, stats_RPE_rot, stats_RPE_full]
            
            if save:
                saveARPE2csv(type_traj, "RPE", stats_RPE, sequence) 
            return stats_RPE            


###

# title_1 = "Alignment of the camera trajectory before BA with the ground truth: " + sequence
# title_ = "Aligned camera trajectory before BA: " + sequence