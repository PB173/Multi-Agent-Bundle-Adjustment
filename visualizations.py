# This file contains all the functions that are used to visualize results of the MABA master thesis.
import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation as animate

def plot_trajectories(keyframes, labels):
    cam_tran = []
    time = []
    for i in range(len(keyframes)):
        kf = keyframes[i]
        np_camera_translations = np.reshape([np.array(keyframe[2:5]) for keyframe in kf], (-1, 3))
        t = [keyframe[1].astype(float) for keyframe in kf]
        cam_tran.append(np_camera_translations)
        time.append(t)

    fig_1, ax_1 = plt.subplots(3, 1, figsize=(32,18))
    fig_1.subplots_adjust(left=0.07, bottom=0.08, right = 0.96, top=0.96, hspace = 0.40)

    fig_2, ax_2 = plt.subplots(3, 1, figsize= (32,18))
    fig_2.subplots_adjust(left=0.07, bottom=0.08, right = 0.96, top=0.96, hspace = 0.40)
    
    fig_2d, ax_2d = plt.subplots(1,1, figsize= (32,18))
    fig_2d.subplots_adjust(left=0.07, bottom=0.08, right = 0.96, top=0.96)

    fig_3d = plt.figure(figsize= (32,18))
    ax_3d = fig_3d.add_subplot(projection='3d')
    fig_3d.subplots_adjust(left=0.07, bottom=0.08, right = 0.96, top=0.96)

    for i in range(len(cam_tran)): # Adapt loop to go through all agents provided for the plot
        x_coor = [camera_translation[0] for camera_translation in cam_tran[i]]
        y_coor = [camera_translation[1] for camera_translation in cam_tran[i]]
        z_coor = [camera_translation[2] for camera_translation in cam_tran[i]]

        ax_1[0].scatter(time[i], x_coor, linewidths = 3, marker ="o",s = 14, label = labels[i])
        ax_1[1].scatter(time[i], y_coor, linewidths = 3, marker ="o",s = 14, label = labels[i])
        ax_1[2].scatter(time[i], z_coor, linewidths = 3, marker ="o",s = 14, label = labels[i])
        
        ax_2[0].scatter(range(len(x_coor)), x_coor, linewidths = 3, marker ="o",s = 14, label = labels[i])
        ax_2[1].scatter(range(len(y_coor)), y_coor, linewidths = 3, marker ="o",s = 14, label = labels[i])
        ax_2[2].scatter(range(len(z_coor)), z_coor, linewidths = 3, marker ="o",s = 14, label = labels[i])
                
        ax_2d.scatter(x_coor, y_coor, linewidths = 3, marker ="o",s = 14, label = labels[i])

        ax_3d.scatter(x_coor, y_coor, z_coor, marker='o', label = labels[i])

    ax_1[0].set_title('x position', fontsize = 30, fontweight ='bold')
    ax_1[1].set_title('y position', fontsize = 30, fontweight ='bold')
    ax_1[2].set_title('z position', fontsize = 30, fontweight ='bold')  

    ax_2[0].set_title('x position', fontsize = 30, fontweight ='bold')
    ax_2[1].set_title('y position', fontsize = 30, fontweight ='bold')
    ax_2[2].set_title('z position', fontsize = 30, fontweight ='bold') 
    
    for axis in ax_1:
        axis.set_xlabel('Time [s]', fontsize = 24), axis.set_ylabel('Position', fontsize = 24)
        axis.legend(fontsize = 24, markerscale = 2)
        axis.minorticks_on(), axis.ticklabel_format(useOffset=False)
        axis.tick_params(axis='both', labelsize=18)
        axis.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
        axis.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')
    
    for axis in ax_2:
        axis.set_xlabel('Keyframe', fontsize = 24), axis.set_ylabel('Position', fontsize = 24)
        axis.legend(fontsize = 24, markerscale = 2)
        axis.minorticks_on(), axis.ticklabel_format(useOffset=False)
        axis.tick_params(axis='both', labelsize=18)
        axis.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
        axis.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')
        axis.set_xlim([0, 275])
        

    ax_2d.set_title('XY position', fontsize = 30, fontweight ='bold')
    ax_2d.set_xlabel('x position', fontsize = 24), ax_2d.set_ylabel('y position', fontsize = 24)
    ax_2d.legend(fontsize = 24, markerscale = 2)
    ax_2d.minorticks_on()
    ax_2d.tick_params(axis='both', labelsize=18)
    ax_2d.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    ax_2d.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')

    ax_3d.set_xlabel('\nx position',fontsize = 24, fontweight ='bold', linespacing=3.2)
    ax_3d.set_ylabel('\ny position',fontsize = 24, fontweight ='bold', linespacing=3.2)
    ax_3d.set_zlabel('\nz position',fontsize = 24, fontweight ='bold', linespacing=3.2)
    ax_3d.set_title('3D trajectories', fontsize = 30, fontweight ='bold')
    ax_3d.tick_params(axis='both', labelsize=18)
    ax_3d.legend(fontsize = 24, markerscale = 2)

    plt.show()

def compare_trajectories(keyframes, traj_cor, ORB_version, idx, save):
    # Create array containing trajectory before BA
    traj_org = np.reshape([np.array(keyframe[2:5]) for keyframe in keyframes], (-1, 3))

    # Initialize figures
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(32, 18)
    fig.subplots_adjust(top=0.96, bottom=0.06, left=0.06, right=0.99, hspace=0.37, wspace=0.2)

    # Initialize variables & plot
    x_org = [camera_translation[0] for camera_translation in traj_org]
    y_org = [camera_translation[1] for camera_translation in traj_org]
    z_org = [camera_translation[2] for camera_translation in traj_org]

    x_cor = [camera_translation[0] for camera_translation in traj_cor]
    y_cor = [camera_translation[1] for camera_translation in traj_cor]
    z_cor = [camera_translation[2] for camera_translation in traj_cor]

    ax[0].scatter(range(len(x_org)), x_org, linewidths = 3, marker ="o",s = 14,
                  label = 'Before BA')
    ax[1].scatter(range(len(y_org)), y_org, linewidths = 3, marker ="o",s = 14,
                  label = 'Before BA')
    ax[2].scatter(range(len(z_org)), z_org, linewidths = 3, marker ="o",s = 14,
                  label = 'Before BA')

    ax[0].scatter(range(len(x_cor)), x_cor, linewidths = 3, marker ="o",s = 14,
                  label = 'After BA')
    ax[1].scatter(range(len(y_cor)), y_cor, linewidths = 3, marker ="o",s = 14,
                  label = 'After BA')
    ax[2].scatter(range(len(z_cor)), z_cor, linewidths = 3, marker ="o",s = 14,
                  label = 'After BA')
            
    ax[0].set_title('x position', fontsize = 30, fontweight ='bold')
    ax[1].set_title('y position', fontsize = 30, fontweight ='bold')
    ax[2].set_title('z position', fontsize = 30, fontweight ='bold') 

    for axis in ax:
        axis.set_xlabel('Keyframe', fontsize = 24), axis.set_ylabel('Position', fontsize = 24)
        axis.legend(fontsize = 24, markerscale=2)
        axis.minorticks_on(), axis.ticklabel_format(useOffset=False)
        axis.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
        # axis.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')
        axis.set_xlim([0, len(traj_org)+50])
        axis.tick_params(axis='both', which='major', labelsize=24)
    if save:
        plt.savefig("/home/pb/Documents/Thesis/Figures/ORB SLAM/Camera Trajectories/After BA/Inverse direct method/{}/MH0{}.svg".format(ORB_version, idx), 
                    bbox_inches = 'tight', pad_inches = 0.2, transparent = False)

    fig_3d = plt.figure(figsize = (10, 7))
    ax_3d = fig_3d.add_subplot(projection='3d')

    ax_3d.scatter(x_org, y_org, z_org, marker='o', label = 'Before BA')
    ax_3d.scatter(x_cor, y_cor, z_cor, marker='o', label = 'After BA')

    ax_3d.set_xlabel('x position',fontsize = 24, fontweight ='bold')
    ax_3d.set_ylabel('y position',fontsize = 24, fontweight ='bold')
    ax_3d.set_zlabel('z position',fontsize = 24, fontweight ='bold')
    ax_3d.set_title('Resulting 3D trajectories after applying Bundle Adjustment', fontsize = 14, fontweight ='bold', y=1.1)
    ax_3d.legend(fontsize = 14)

    plt.show()

def compare_noisy_trajectories(traj_org, traj_noisy, traj_BA, traj_BA_smooth, ground_truth, timestamps_kf, sequence, save, save_fig_path):

    # Initialize figures
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(32, 18)
    # fig.subplots_adjust(top=0.96, bottom=0.06, left=0.06, right=0.99, hspace=0.45, wspace=0.2)
    fig.subplots_adjust(top=0.93, bottom=0.075, left=0.07, right=0.99, hspace=0.45, wspace=0.2)

    # Initialize variables & plot
    x_org = [camera_translation[0] for camera_translation in traj_org]
    y_org = [camera_translation[1] for camera_translation in traj_org]
    z_org = [camera_translation[2] for camera_translation in traj_org]

    x_noisy = [camera_translation[0] for camera_translation in traj_noisy]
    y_noisy = [camera_translation[1] for camera_translation in traj_noisy]
    z_noisy = [camera_translation[2] for camera_translation in traj_noisy]

    x_BA = [camera_translation[0] for camera_translation in traj_BA]
    y_BA = [camera_translation[1] for camera_translation in traj_BA]
    z_BA = [camera_translation[2] for camera_translation in traj_BA]

    if np.any(ground_truth):
        x_gt = np.array(ground_truth[" p_RS_R_x [m]"])
        y_gt = np.array(ground_truth[" p_RS_R_y [m]"])
        z_gt = np.array(ground_truth[" p_RS_R_z [m]"])
        t_gt = np.array(ground_truth["#timestamp"])*1e-9

        # gt_rot_w = np.array(ground_truth[" q_RS_w []"])
        # gt_rot_x = np.array(ground_truth[" q_RS_x []"])
        # gt_rot_y = np.array(ground_truth[" q_RS_y []"])
        # gt_rot_z = np.array(ground_truth[" q_RS_z []"])

        t_idx = []
        for time in timestamps_kf:
            t_idx.append(np.argmin(np.abs(time - t_gt)))

        x_gt_plot = x_gt[t_idx]
        y_gt_plot = y_gt[t_idx]
        z_gt_plot = z_gt[t_idx]

    # BA_traj = [x_BA, y_BA, z_BA]
    # gt_traj = [x_gt, y_gt, z_gt]
    # np.save('/home/pb/Documents/Thesis/gt_traj.npy', gt_traj)
    # np.save('/home/pb/Documents/Thesis/BA_traj.npy', BA_traj)

    # gt_rot = [gt_rot_w, gt_rot_x, gt_rot_y, gt_rot_z]
    # np.save('/home/pb/Documents/Thesis/gt_cam_rot.npy', gt_rot)


    if traj_BA_smooth:
        x_BA_smooth = [camera_translation[0] for camera_translation in traj_BA_smooth]
        y_BA_smooth = [camera_translation[1] for camera_translation in traj_BA_smooth]
        z_BA_smooth = [camera_translation[2] for camera_translation in traj_BA_smooth]


    ax[0].scatter(range(len(x_org)), x_org, linewidths = 3, marker = "x", color = "black", s = 150,
                  label = 'Original camera position')
    ax[1].scatter(range(len(y_org)), y_org, linewidths = 3, marker = "x", color = "black", s = 150,
                  label = 'Original camera position')
    ax[2].scatter(range(len(z_org)), z_org, linewidths = 3, marker = "x", color = "black", s = 150,
                  label = 'Original camera position')

    # ax[0].scatter(range(len(x_noisy)), x_noisy, linewidths = 3, marker = "o", color = "red", s = 50,
    #               label = 'Noisy camera position')
    # ax[1].scatter(range(len(y_noisy)), y_noisy, linewidths = 3, marker = "o", color = "red", s = 50,
    #               label = 'Noisy camera position')
    # ax[2].scatter(range(len(z_noisy)), z_noisy, linewidths = 3, marker = "o", color = "red", s = 50,
    #               label = 'Noisy camera position')

    # ax[0].scatter(range(len(x_BA)), x_BA, linewidths = 3, marker = "^", c = "green", s = 50,
    #               label = 'After BA')
    # ax[1].scatter(range(len(y_BA)), y_BA, linewidths = 3, marker = "^", c = "green", s = 50,
    #               label = 'After BA')
    # ax[2].scatter(range(len(z_BA)), z_BA, linewidths = 3, marker = "^", c = "green", s = 50,
    #               label = 'After BA')
    
    # if np.any(ground_truth):
    #     ax[0].scatter(range(len(x_gt_plot)), x_gt_plot, linewidths = 3, marker = "^", c = "yellow", s = 50,
    #                 label = 'Ground truth')
    #     ax[1].scatter(range(len(y_gt_plot)), y_gt_plot, linewidths = 3, marker = "^", c = "yellow", s = 50,
    #                 label = 'Ground truth')
    #     ax[2].scatter(range(len(z_gt_plot)), z_gt_plot, linewidths = 3, marker = "^", c = "yellow", s = 50,
    #                 label = 'Ground truth')
    
    # if traj_BA_smooth:
    #     ax[0].scatter(range(len(x_BA_smooth)), x_BA_smooth, linewidths = 3, marker = "s", c = "cyan", s = 2,
    #                   label = 'After BA (smooth)')
    #     ax[1].scatter(range(len(y_BA_smooth)), y_BA_smooth, linewidths = 3, marker = "s", c = "cyan", s = 2,
    #                   label = 'After BA (smooth)')
    #     ax[2].scatter(range(len(z_BA_smooth)), z_BA_smooth, linewidths = 3, marker = "s", c = "cyan", s = 2,
    #                   label = 'After BA (smooth)')
            
    # ax[0].set_title('X position', fontsize = 30, fontweight ='bold')
    # ax[1].set_title('Y position', fontsize = 30, fontweight ='bold')
    # ax[2].set_title('Z position', fontsize = 30, fontweight ='bold') 


    labels = ['X position', 'Y position', 'Z position']
    for i, axis in enumerate(ax):
        axis.set_xlabel('Keyframe', fontsize = 30), axis.set_ylabel(labels[i], fontsize = 30)
        # axis.legend(fontsize = 24, markerscale=2)
        axis.minorticks_on(), axis.ticklabel_format(useOffset=False)
        axis.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
        # axis.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')
        axis.set_xlim([0, len(traj_org)+5])
        axis.tick_params(axis='both', which='major', labelsize=24)
    
    plt.suptitle("ORB-SLAM2 output trajectory: {}".format(sequence), fontsize = 40, fontweight = 'bold')
    
    if save:
        plt.savefig(save_fig_path + 'camera_poses_2D.png', bbox_inches = 'tight', pad_inches = 0.2, transparent = False)

    fig_3d = plt.figure(figsize = (32, 18))
    ax_3d = fig_3d.add_subplot(projection='3d')

    ax_3d.scatter(x_org, y_org, z_org, marker='o', label = 'Before BA')
    ax_3d.scatter(x_BA, y_BA, z_BA, marker='o', label = 'After BA')

    ax_3d.set_xlabel('\nx position',fontsize = 24, fontweight ='bold')
    ax_3d.set_ylabel('\ny position',fontsize = 24, fontweight ='bold')
    ax_3d.set_zlabel('\nz position',fontsize = 24, fontweight ='bold')
    ax_3d.set_title('Resulting 3D trajectories after applying Bundle Adjustment: {}'.format(sequence), fontsize = 30, fontweight = 'bold', y = 1.1)
    ax_3d.legend(fontsize = 24, markerscale = 2)
    ax_3d.tick_params(axis='both', which='major', labelsize=18)

    if save:
        plt.savefig(save_fig_path + 'camera_poses_3D.png', bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
    
    if not save:
        plt.show()

def plot_pointclouds(keyframes, points, max_nbr_pts_per_kf, animation, save):
    np_point_coors = np.reshape([np.array(point[0]) for point in points], (-1, 3))
    np_camera_translations = np.reshape([np.array(keyframe[2:5]) for keyframe in keyframes], (-1, 3))
    np_frame_ids = [np.array([(point[obs+2][0]).astype(int) for obs in range(len(point)-2)]) for point in points]

    keyframe_id = [keyframe[0].astype(int) for keyframe in keyframes]

    if animation == "dynamic": # create animated figure and axis
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x_data, y_data, z_data = [], [], [] # initialize empty array for points
        
        def update_animation(frame):
            ax.clear()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Keyframe %d' % frame)
            ax.set_xlim((-1,5))
            ax.set_ylim((-1,1))
            ax.set_zlim((-2,3))
            ax.view_init(elev=20, azim=-120)
            ax.scatter(x_data[frame], y_data[frame], z_data[frame])
            ax.scatter([cam[0] for cam in cam_pos[:frame]], [cam[1] for cam in cam_pos[:frame]],
                       [cam[2] for cam in cam_pos[:frame]], marker='^', c='r')
            
    elif animation ==  "static": # create static figure and axis
        fig = plt.figure()
        ax_stat = fig.add_subplot(projection='3d')
        ax_stat.set_xlabel('X')
        ax_stat.set_ylabel('Y')
        ax_stat.set_zlabel('Z')
    else:
         raise ValueError("Invalid choice for input animation. Please choose between 'static' or 'dynamic'.")
                  
    # initialize vectors for plotting
    cam_pos = []
    for i in range(len(keyframe_id)): # len(keyframe_id)
        kf = keyframe_id[i]
        cam_pos.append(np_camera_translations[i])
        point_coor = []
        for j in range(len(np_frame_ids)):
            if kf in np_frame_ids[j]: # Point observed by camera at current keyframe
                point_coor.append(np_point_coors[j]) # array containing coordinates of all observed points
        
        if "static" in animation: # No animation, only the static plot is generated
            # ax_stat.scatter([pt[0] for pt in point_coor[:max_nbr_pts_per_kf]], [pt[1] for pt in point_coor[:max_nbr_pts_per_kf]],
            #            [pt[2] for pt in point_coor[:max_nbr_pts_per_kf]], marker='o',c='b')
            ax_stat.scatter([pt[0] for pt in point_coor[0::100]], [pt[1] for pt in point_coor[0::100]],
                            [pt[2] for pt in point_coor[0::100]], marker='o',c='b')
        
        else: # Only the animated plot is generated
            x_data.append([pt[0] for pt in point_coor[:max_nbr_pts_per_kf]])
            y_data.append([pt[1] for pt in point_coor[:max_nbr_pts_per_kf]])
            z_data.append([pt[2] for pt in point_coor[:max_nbr_pts_per_kf]])

            if i == len(keyframe_id)-1:
                ani = animate(fig, update_animation, frames=len(x_data), interval=150)
                if save:
                    ani.save('../../Figures/ORB SLAM/Camera Trajectories/MH_01_pointcloud.gif', writer='pillow')
    if animation == "static": # print cam pos
        ax_stat.scatter([cam[0] for cam in cam_pos], [cam[1] for cam in cam_pos],[cam[2] for cam in cam_pos], marker='^', c='r')
        # ax_stat.set_xlim([-20, 20]), ax_stat.set_ylim([0, 200])
    plt.show()

def plot_noisy_pointclouds(points_org, points_noisy, points_BA, sequence, save, save_fig_path):
    fig = plt.figure()
    fig.set_size_inches(32, 18)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('\n\n\nx position',fontsize = 30, fontweight ='bold')
    ax.set_ylabel('\n\n\ny position',fontsize = 30, fontweight ='bold')
    ax.set_zlabel('\n\n\nz position',fontsize = 30, fontweight ='bold')
    ax.set_title("Position of points after BA: {}".format(sequence), fontsize = 40, fontweight = 'bold', y = 1.1)
    ax.tick_params(axis='both', which='major', labelsize=24)
         
    # ax.scatter([pt[0] for pt in points_org[0::100]], [pt[1] for pt in points_org[0::100]],
    #                 [pt[2] for pt in points_org[0::100]], marker='o', c='blue')
    
    # ax.scatter([pt[0] for pt in points_noisy[0::100]], [pt[1] for pt in points_noisy[0::100]],
    #                 [pt[2] for pt in points_noisy[0::100]], marker='o', c='orange')
    
    # ax.scatter([pt[0] for pt in points_BA[0::100]], [pt[1] for pt in points_BA[0::100]],
    #                 [pt[2] for pt in points_BA[0::100]], marker='o', c='green')

    ax.scatter([pt[0] for pt in points_org], [pt[1] for pt in points_org],
                    [pt[2] for pt in points_org], marker='x', c='black', s = 100)
    
    ax.scatter([pt[0] for pt in points_noisy], [pt[1] for pt in points_noisy],
                    [pt[2] for pt in points_noisy], marker='o', c='red', s = 100)
    
    ax.scatter([pt[0] for pt in points_BA], [pt[1] for pt in points_BA],
                    [pt[2] for pt in points_BA], marker='^', c='green', s = 50)
    
    labels= ('Original point position', 'Noisy point position', 'After BA')
    ax.legend(labels, fontsize = 24, markerscale=2)

    # ax.set_xlim(-1.5, 4)
    # ax.set_ylim(-1.5, 1)
    # ax.set_zlim(-1.5, 4)

    if save:
        plt.savefig(save_fig_path + 'point_cloud.png', bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
    
    if not save:
        plt.show()

def compare2groundtruth(keyframes, cam_pos_cor, ground_truth, do_plot):
    """
    Calculate difference between an optimized trajectory and the ground truth provided by the leica LiDar. Note that, in order to do this, the estimated trajectory
    and the ground truth should first be matched using Sim3 (rotation, translation, scale). Optional plot of the ground truth and trajectory can be provided.
 
    Input
    :param keyframes: contains the timestamp and original estimates of the camera position
    :param cam_pos_cor: contains the optimized camera positions after applying Bundle Adjustment 
    :param ground_truth: the ground truth data containing the timestamp and x, y and z coordinates
    Output
    :return: A vector containing the differences
    """

    cam_pos_org = np.reshape([np.array(keyframe[2:5]) for keyframe in keyframes], (-1, 3))
    x_org = [camera_translation[0] for camera_translation in cam_pos_org]
    y_org = [camera_translation[1] for camera_translation in cam_pos_org]
    z_org = [camera_translation[2] for camera_translation in cam_pos_org]

    x_cor = [camera_translation[0] for camera_translation in cam_pos_cor]
    y_cor = [camera_translation[1] for camera_translation in cam_pos_cor]
    z_cor = [camera_translation[2] for camera_translation in cam_pos_cor]

    cam_timestamps = [keyframe[1].astype(float) for keyframe in keyframes]
    gt_timestamps = np.array(ground_truth["Timestamps"]*1e-9)
    gt_x, gt_y, gt_z = np.array(ground_truth["p_RS_R_x"]), np.array(ground_truth["p_RS_R_y"]), np.array(ground_truth["p_RS_R_z"])
    norm_org, norm_cor = [], []
    gt_x_kf, gt_y_kf, gt_z_kf = [], [], []
    for i in range(len(cam_timestamps)):
        # align estimated trajectory to ground truth
        idx = np.where(np.abs(gt_timestamps - cam_timestamps[i]) == np.abs(gt_timestamps - cam_timestamps[i]).min())[0][0]
        gt_x_kf.append(gt_x[idx]), gt_y_kf.append(gt_y[idx]), gt_z_kf.append(gt_z[idx])

        # calculate difference between traj and ground truth in 3 position coordinated 
        diff_x_org, diff_y_org, diff_z_org = x_org[i] - gt_x[idx], y_org[i] - gt_y[idx], z_org[i] - gt_z[idx]
        diff_x_cor, diff_y_cor, diff_z_cor = x_cor[i] - gt_x[idx], y_cor[i] - gt_y[idx], z_cor[i] - gt_z[idx]

        # calculate L2 norm (SSE) for differences over 3 dimensions
        norm_org.append(norm([diff_x_org, diff_y_org, diff_z_org]))
        norm_cor.append(norm([diff_x_cor, diff_y_cor, diff_z_cor]))
    
    # resulting difference is the average L2 norm over all timesteps
    avg_norm_org =  sum(norm_org)/len(norm_org)
    avg_norm_cor =  sum(norm_cor)/len(norm_cor)
    
    print("Average L2 norm of Before BA:", avg_norm_org)
    print("Average L2 norm after BA:", avg_norm_cor)

    if do_plot:
        fig, ax = plt.subplots(3, 1)
        fig.tight_layout()
        
        ax[0].scatter(range(len(x_org)), x_org, linewidths = 2, marker = "o",s = 10, label = 'Estimations')
        ax[0].scatter(range(len(gt_x_kf)), gt_x_kf, linewidths = 2, marker = "_", s = 10, c='r',label = 'Ground Truth')

        ax[1].scatter(range(len(y_org)), y_org, linewidths = 2, marker = "o",s = 10, label = 'Estimations')
        ax[1].scatter(range(len(gt_y_kf)), gt_y_kf, linewidths = 2, marker = "_", s = 10, c='r',label = 'Ground Truth')

        ax[2].scatter(range(len(z_org)), z_org, linewidths = 2, marker = "o",s = 10, label = 'Estimations')
        ax[2].scatter(range(len(gt_z_kf)), gt_z_kf, linewidths = 2, marker = "_", s = 10, c='r',label = 'Ground Truth')

        ax[0].set_title('x position', fontsize = 24, fontweight ='bold')
        ax[1].set_title('y position', fontsize = 24, fontweight ='bold')
        ax[2].set_title('z position', fontsize = 24, fontweight ='bold')  
        
        for axis in ax:
            axis.set_xlabel('Keyframe', fontsize = 12), axis.set_ylabel('Position', fontsize = 12)
            axis.legend(fontsize = 14)
            axis.minorticks_on(), axis.ticklabel_format(useOffset=False)
            axis.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
            axis.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')
            axis.set_xlim([0, len(x_org)+30])

        plt.show()

def visualize_groundtruth(ground_truth, xlim, ylim,  zlim):
    # 2D figure: x-y
    fig_2D = plt.figure(figsize=(9, 9))
    for i in range(len(ground_truth)):
        gt_x, gt_y = np.array(ground_truth[i]["p_RS_R_x"]), np.array(ground_truth[i]["p_RS_R_y"])
        plt.plot(gt_x, gt_y, linewidth = 1,label = 'MH0'+str(i+1))

    plt.xlabel('x [m]', fontsize = 14), plt.ylabel('y [m]', fontsize = 14)
    plt.xlim(xlim), plt.ylim(ylim)
    plt.title("Ground Truth EuRoC dataset: 2D", fontsize = 16, fontweight = "bold")
    plt.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    plt.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')
    plt.legend(fontsize = 14)
    plt.savefig("/home/pb/Documents/Thesis/Figures/ORB SLAM/Ground Truth/ground_truth_2D.svg", 
            bbox_inches = 'tight', pad_inches = 0.2)

    # 3D figure
    fig_3D = plt.figure(figsize=(9, 9))
    ax_3D = plt.axes(projection="3d")

    for i in range(len(ground_truth)):
        gt_x, gt_y = np.array(ground_truth[i]["p_RS_R_x"]), np.array(ground_truth[i]["p_RS_R_y"])
        gt_z = np.array(ground_truth[i]["p_RS_R_z"])
    
        ax_3D.plot3D(gt_x, gt_y, gt_z, label = 'MH0'+str(i+1))
    
    ax_3D.set_xlim(xlim), ax_3D.set_ylim(ylim), ax_3D.set_zlim(zlim)
    ax_3D.set_xlabel('x [m]', fontsize = 14), ax_3D.set_ylabel('y [m]', fontsize = 14)
    ax_3D.set_zlabel('z [m]', fontsize = 14)
    ax_3D.set_title("Ground Truth EuRoC dataset: 3D", pad=50, fontsize = 16, fontweight = "bold")
    ax_3D.legend(fontsize = 14)
    plt.savefig("/home/pb/Documents/Thesis/Figures/ORB SLAM/Ground Truth/ground_truth_3D.svg", 
            bbox_inches = 'tight', pad_inches = 0.2)


    plt.show()

def visualize_observations(trajectory, ground_truth, xlim, ylim):

    gt_x, gt_y= np.array(ground_truth["p_RS_R_x"]), np.array(ground_truth["p_RS_R_y"])
    
    plt.figure(figsize=(9, 9))
    for i in range(len(trajectory)):
        x = [camera_translation[0] for camera_translation in trajectory[i]]
        y = [camera_translation[1] for camera_translation in trajectory[i]]
        plt.plot(x,y, linewidth = 1, c = 'k',label = 'Observations' + str(i+1))
    
    plt.plot(gt_x, gt_y, linewidth = 1, c = 'r',label = 'Ground Truth')
    
    plt.xlabel('x [m]'), plt.ylabel('y [m]')
    plt.xlim(xlim), plt.ylim(ylim)
    plt.title("Ground truth")

    plt.legend()
    plt.show()
