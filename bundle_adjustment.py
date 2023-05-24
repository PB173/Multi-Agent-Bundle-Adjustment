##################### Import modules and functions #####################
PyCeres_location = "/home/pb/ceres-bin/lib"
import sys
sys.path.insert(0,PyCeres_location)
import PyCeres as ceres # Import the Python Bindings

lib_location = "/home/pb/Documents/Thesis/Scripts/lib"
sys.path.insert(0, lib_location)

from project_keyframe import inverse_transformation_matrix, inverse_transform_points
from rotation_transformations import quat2rotmat, rotmat2quat
from functions import information_points, information_keyframes
from Visualizations.visualizations import compare_noisy_trajectories, plot_noisy_pointclouds
import numpy as np

import matplotlib.pyplot as plt
import copy
import math

from scipy import interpolate
# from sklearn.metrics.pairwise import cosine_similarity


## Suppress warnings
import warnings

def fxn():
    warnings.warn("runtime", RuntimeWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

##################### Helper functions General #####################

# Check number of keyframes
def check_nbr_kf(points):
    """
    Checks how many keyframes observe a certain point. This knowledge can be used to discard points seen 
    by too few keyframes.

    Arguments:
    - points -- Point cloud observed by a specific sequence (or a combination of sequences)

    Returns:
    - nbr_kf_per_point -- For every point the number of keyframes that observe it
    """
    # Extract data from provided keyframes and points
    numpy_frame_ids = [np.array([(point[obs+2][0]).astype(int) for obs in range(len(point)-2)]) for point in points]

    nbr_kf_per_point = []
    for i in range(len(points)): # loop over all the points in the provided pointcloud
        frame_ids = numpy_frame_ids[i]
        nbr_kf_per_point.append(len(frame_ids))

    return nbr_kf_per_point

# Implement random walk noise
def random_walk_noise(BA_algorithm, cam_positions, noise_factor, threshold=10, split_seq = 0):
    """"
    Implement random walk noise on camera positions to simulate error in camera measurements.

    Arguments:
    - BA_algorithm: 'SABA' or 'MABA' to indicate which version of the algortihm is considered
    - cam_positions: List containing the positions of the camera center
    - noise factor: Intensity of the noise
    - threshold: Frame from which noise is applied (default = 10)
    - split_seq: Indicates the length of the first sequence when the MABA algorithm is considered (default = 0 for SABA)

    Returns:
    - cam_positions_noise: Noisy camera positions
    """
    # Initialize trajectory variables
    x = [traj[0] for traj in cam_positions]
    y = [traj[1] for traj in cam_positions]
    z = [traj[2] for traj in cam_positions]
    
    cam_positions_noise = []

    # Generate random walk noise
    if BA_algorithm == 'SABA':
        # Initialize noise array
        num_steps = len(x)
        noise = np.zeros((3, num_steps))
        # cam_positions_noise = []
        for i in range(threshold, num_steps):
            noise[0, i] = noise[0, i-1] + np.random.normal(noise_factor, 0.001)
            noise[1, i] = noise[1, i-1] + np.random.normal(noise_factor, 0.001)
            noise[2, i] = noise[2, i-1] + np.random.normal(noise_factor, 0.001)    
            
            # Add the noise to the trajectory
        x_noisy = x + noise[0]
        y_noisy = y + noise[1]
        z_noisy = z + noise[2]
    
    elif BA_algorithm == 'MABA':
        number_steps = [split_seq, len(x)-split_seq] # Split in two before adding noise to avoid over-acummulating effect for second sequence

        noise_total_x, noise_total_y, noise_total_z = [], [], []
        for i, num_steps in enumerate(number_steps):
            noise = np.zeros((3, num_steps))
            for j in range(threshold, num_steps):
                noise[0, j] = noise[0, j-1] + np.random.normal(noise_factor, 0.001)
                noise[1, j] = noise[1, j-1] + np.random.normal(noise_factor, 0.001)
                noise[2, j] = noise[2, j-1] + np.random.normal(noise_factor, 0.001)    
          
            noise_total_x.append(noise[0]), noise_total_y.append(noise[1]), noise_total_z.append(noise[2])

        noise_tot_x, noise_tot_y, noise_tot_z = np.concatenate((noise_total_x[0], noise_total_x[1])),np.concatenate((noise_total_y[0], noise_total_y[1])), np.concatenate((noise_total_z[0], noise_total_z[1]))

            # Add the noise to the trajectory
        x_noisy = x + noise_tot_x
        y_noisy = y + noise_tot_y
        z_noisy = z + noise_tot_z

    # Merge data in 1 output vector
    for i in range(len(x_noisy)):
        vec = np.array([x_noisy[i], y_noisy[i], z_noisy[i]])
        cam_positions_noise.append(vec)

    return cam_positions_noise

# Function to deliberately add noise to the estimates (points + camera position)
def add_noise_to_ORB_output(noise_type, BA_algorithm, points, cam_positions, noise_factor, threshold=10, split_seq = 0):
    """
    Apply noise to camera positions and/or points to simulate erronous measurements.

    Arguments:
    - noise_type (str): Specify which noise is added (options: 'Gaussian', 'Uniform', or 'Random Walk').
    - BA_algorithm (str): Bundle adjustment algorithm used (options: 'MABA' or 'SABA').
    - points (list): Point cloud observed by a specific sequence (or a combination of sequences).
    - cam_positions (list): List containing the positions of the camera center.
    - noise_factor (float): Intensity of the noise.
    - threshold (int, optional): Frame from which noise is applied (default = 10).
    - split_seq (int, optional): Indicates the length of the first sequence when the MABA algorithm is considered (default = 0 for SABA).

    Returns:
    - cam_pos_noise (list): List containing the noisy positions of the camera center.
    - point_coors_noise (list): Noisy point cloud observed by a specific sequence (or a combination of sequences).
    """
    # Initialize variables
    cam_pos_noise = []
    point_coors_noise = []
    point_coors = np.reshape([np.array(point[0]) for point in points], (-1, 3))

    frame_ids = [np.array([(point[obs+2][0]).astype(int) for obs in range(len(point)-2)]) for point in points]

    # Add noise to camera position
    if noise_type == 'Random Walk':
        cam_pos_noise = random_walk_noise(BA_algorithm, cam_positions, noise_factor, threshold, split_seq)

    else:
        for i, keyframe in enumerate(cam_positions):
            if i >= threshold: #
                if noise_type == 'Gaussian':
                    keyframe_noise = keyframe + np.random.normal(0, noise_factor, 3)
                elif noise_type == 'Uniform':
                    keyframe_noise = keyframe + np.random.uniform(low = -2*noise_factor, high = 2*noise_factor, size = 3)
                elif noise_type == 'None':
                    keyframe_noise = keyframe

            else:
                keyframe_noise = keyframe
            cam_pos_noise.append(keyframe_noise)

    # Add noise to points
    for i, point in enumerate(point_coors):
        if np.min(frame_ids[i]) >= threshold:
            if noise_type == 'Gaussian' or noise_type == 'Random Walk':
                point_noise = point + np.random.normal(0, noise_factor, 3)
            elif noise_type == 'Uniform':
                point_noise = point + np.random.uniform(low = -2*noise_factor, high = 2*noise_factor, size = 3)
            elif noise_type == 'None':
                point_noise = point
            else:
                point_noise = point
            # point_noise = point
        else:
            point_noise = point
        point_coors_noise.append(point_noise)
        
    return cam_pos_noise, point_coors_noise

# Visualize the accumulated cost at every iteration
def visualize_cost(cost_iter, save = 0, save_path = ''):
    """
    Visualize accumulated cost at every iteration of the solution process. Data obtained from Ceres non-linear solver.

    Arguments:
    - cost_iter: Accumulated cost at every iteration
    - save: Boolean; indicate whether to save the plot (default = 0)
    - save_path: Path for saving figure

    Returns:
    - Plot of cost function

    """
    fig, ax = plt.subplots(1,1)
    fig.subplots_adjust(left=0.14, bottom = 0.14, top = 0.88, right = 0.9)
    ax.plot(cost_iter, 'bx--')
    ax.set_xlabel('Iteration', fontsize = 20)
    ax.set_ylabel('Cost',fontsize = 20)
    ax.set_yscale('log')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_title('Evaluation of the absolute cost',fontweight ='bold', fontsize= 24, y= 1.05)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')

    if save:
        plt.savefig(save_path + 'cost.png', bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
        plt.close('all')
    else:
        plt.show()

# Detect obvious outliers in the BA trajectory
def detect_outliers(coor, constant = 0.5, threshold=1.1):
    """
    Detect and smoothen outliers in camera trajectory after BA.

    Arguments:
    - coor: Array containing the camera position coordinates in 1 direction
    - constant: The considered constant value in the calculation of the rejection ratio (default = 0.5)
    - threshold: Threshold for which a point is considered an outlier (default = 1.1)

    Returns:
    - x: Array containing the indices of the cameras
    - coor: Array containing the smoothed camera coordinates in 1 direction

    """
    x = np.linspace(0, len(coor)-1, len(coor))
    # First camera pose
    idx = 0
    if (abs(coor[idx] + constant))/(abs(coor[idx+1]) + constant) > threshold and abs(coor[idx]) > 0.01: # upper outlier
        coor[idx] = np.mean([coor[idx], coor[idx+1]])
    elif (abs(coor[idx+1]) + constant)/(abs(coor[idx]) + constant) > threshold and abs(coor[idx]) > 0.01: # lower outlier
        coor[idx] = np.mean([coor[idx], coor[idx+1]])
   
    # Camera 2 -- n-1
    for idx in range(1, len(coor)-2):
        # if abs(coor[idx-1]) > 1e-2 and abs(coor[idx+1]) > 1e-2:
        if (abs(coor[idx]) + constant)/(abs(coor[idx-1]) + constant) > threshold and (abs(coor[idx]+0.5))/(abs(coor[idx+1])+0.5) > threshold and abs(coor[idx]) > 0.01: # upper outlier
            coor[idx] = np.mean([coor[idx-1], coor[idx+1]])

        elif (abs(coor[idx-1]) + constant)/(abs(coor[idx]) + constant) > threshold and (abs(coor[idx+1])+0.5)/(abs(coor[idx])+0.5) > threshold and abs(coor[idx]) > 0.01: # lower outlier
            coor[idx] = np.mean([coor[idx-1], coor[idx+1]])
    
    # Last camera pose
    idx = len(coor)-1
    if (abs(coor[idx] + constant))/(abs(coor[idx-1]) + constant) > threshold and abs(coor[idx]) > 0.01: # upper outlier
        coor[idx] = np.mean([coor[idx-1], coor[idx]])
    elif (abs(coor[idx-1]) + constant)/(abs(coor[idx]) + constant) > threshold and abs(coor[idx]) > 0.01: # lower outlier
        coor[idx] = np.mean([coor[idx-1], coor[idx]])
   
    return x, coor

def smoothen_trajectory(traj, constant = 0.5, threshold = 1.1):
    """
    Apply smoothing (outlier filtering) to trajectory to reject obvious outliers.

    Arguments:
    - traj: Trajectory (x, y, z) to which smoothing filter is applied
    - constant: The considered constant value in the calculation of the rejection ratio (default = 0.5)
    - threshold: Threshold for which a point is considered an outlier (default = 1.1)

    Returns:
    - smooth_data: Smoothed trajectory
    """
    smooth_data = []
    x = [camera_translation[0] for camera_translation in traj]
    y = [camera_translation[1] for camera_translation in traj]
    z = [camera_translation[2] for camera_translation in traj]

    _, x_smooth = detect_outliers(x, constant, threshold)
    _, y_smooth = detect_outliers(y, constant, threshold)
    _, z_smooth = detect_outliers(z, constant, threshold)
    
    for idx in range(len(x)):
        smooth_data.append([x_smooth[idx], y_smooth[idx], z_smooth[idx]])

    return smooth_data

def sensor2body(T_BS, cam_pos, cam_rot_mat, point_coor):
    """
    Transform data from sensor frame S to body frame B. This function is typically used when dealing with datasets, such as the EuRoC dataset,
    where the data captured by the different sensors are not expressed in the same reference frame.

    Arguments:
    - T_BS (np.ndarray): Transformation matrix to transform data from sensor frame to body frame.
    - cam_pos (list): List of camera positions. Each element is a numpy array of shape (3,) representing the 3D position of a camera.
    - cam_rot_mat (list): List of camera rotation matrices. Each element is a numpy array of shape (3, 3) representing the rotation matrix.
    - point_coor (list): List of point coordinates. Each element is a numpy array of shape (3,) representing the 3D coordinates of a point.

    Returns:
    - cam_pos_tf (list): Transformed camera positions.
    - cam_rot_vec_tf (list): Transformed camera rotation matrices.
    - point_coor_tf (list): Transformed point coordinates.
    """
    R_BS = T_BS[:3,:3]


    cam_pos_tf, cam_rot_vec_tf = [], []
    for i in range(len(cam_pos)): 
        R = cam_rot_mat[i]
        R_tf = np.matmul(R, R_BS)

        cam_t = np.reshape(cam_pos[i], (-1,3))[0]  # undo effects of transformation on camera position
        cam_R = np.reshape(R_tf[:3,:3], (-1,9))[0] # extract rotation vector

        cam_pos_tf.append(cam_t), cam_rot_vec_tf.append(cam_R)
    
    point_coor_tf = []
    for j in range(len(point_coor)):
        point_coor_tf.append(point_coor[j])

    return cam_pos_tf, cam_rot_vec_tf, point_coor_tf

# Create Ceres problem and add residual blocks
def residualblocks(frame_ids, kf_identifier, point_coors, observations, cam_pos, quat, cam_parameters, th_nbr_kf, BA_type, limit, start_kf=0):
    """
    Create the residual blocks that are fed to the Ceres optimization problem.

    Arguments:
    - frame_ids (list): List of frame IDs corresponding to each point in the point cloud.
    - kf_identifier (list): List of frame IDs for identifying the keyframes.
    - point_coors (list): List of point coordinates. Each element is a numpy array of shape (3,) representing the 3D coordinates of a point.
    - observations (list): List of observations.  Each observation is represented as a tuple (u, v) indicating the 2D image coordinates.
    - cam_pos (list): List of camera positions. Each element is a numpy array of shape (3,) representing the 3D position of a camera.
    - quat (list): List of camera rotations (quaternions). Each element is a numpy array of shape (4,) representing the quaternion parameters.
    - cam_parameters (list): List of camera intrinsic parameters. It contains the following elements:
                             [fx, fy, cx, cy] representing the focal length (in pixels) and principal point coordinates of the camera.
    - th_nbr_kf (int): Threshold number of keyframes. Points seen by fewer keyframes than this threshold are discarded.
    - BA_type (str): Bundle adjustment type. Valid options are: 'motion-only', 'point-only', 'complete', and 'fixed_scale'.
    - limit (str): Parameter limit type. Valid options are: 'Limited' and 'No limits'.
    - start_kf (int, optional): Start keyframe index. Points observed before this keyframe index are ignored (default = 0).

    Returns:
    - problem: Ceres problem object.
    - cnt_ret_pts: Number of points retained for the optimization.
    - discarded_pts: Indices of points that are not considered for the optimization.
    """
    # Create PyCeres problem and define loss function
    problem = ceres.Problem()
    loss_function = ceres.HuberLoss(1.0) 

    # Add point coordinates as parameter blocks
    for i in range(len(point_coors)):
        problem.AddParameterBlock(point_coors[i], 3)
    
    # Add camera positions as parameter blocks
    for i in range(len(cam_pos)):
        problem.AddParameterBlock(cam_pos[i], 3)

    # Add camera rotations (quaternions) as parameter blocks
    for i in range(len(quat)):
        problem.AddParameterBlock(quat[i], 4)

    # Initialize camera intrinsics
    fx, fy, cx, cy = cam_parameters[0], cam_parameters[1], cam_parameters[2], cam_parameters[3]

    # Add the residual terms => calculate reprojection error using CERES
    cnt_ret_pts, discarded_pts = 0, []
    for i in range(len(point_coors)): # loop over all the points in the provided pointcloud
        frame_ids_pt = frame_ids[i]
        Pt_Coor = point_coors[i]

        if len(frame_ids_pt) > th_nbr_kf: # point seen by at least x keyframes
            cnt_ret_pts += 1
            flag = 0
            for j in range(len(frame_ids_pt)): # for every point, loop over the keyframes that observe the point
                if frame_ids_pt[j] > start_kf:
                    idx = kf_identifier.index(frame_ids_pt[j]) # Find index to which frame id corresponds

                    # Define cost function
                    if BA_type == 'motion-only': # Refine only camera poses
                        cost_function = ceres.CostFunction_MotionOnly_fixed(observations[i][j][0], observations[i][j][1], Pt_Coor[0], Pt_Coor[1], Pt_Coor[2], fx, fy, cx, cy)
                        
                        # Add residual block to Ceres problem
                        problem.AddResidualBlock(cost_function, loss_function, quat[idx], cam_pos[idx])
                        flag = 1
                
                    elif BA_type == 'point-only':
                        cost_function = ceres.CostFunction_PointsOnly(observations[i][j][0], observations[i][j][1], fx, fy, cx, cy, cam_pos[idx][0], cam_pos[idx][1], cam_pos[idx][2], \
                                                                        quat[idx][0], quat[idx][1], quat[idx][2], quat[idx][3])
                        if np.any(cam_pos[idx]):
                            pass
                            
                        problem.AddResidualBlock(cost_function, loss_function, point_coors[i])
                        flag = 1

                    elif BA_type == 'complete':
                        cost_function = ceres.CostFunction_FullBA(observations[i][j][0], observations[i][j][1], fx, fy, cx, cy)

                        # Add residual block to Ceres problem
                        problem.AddResidualBlock(cost_function, loss_function, quat[idx], cam_pos[idx], point_coors[i])
                        flag = 1
                    
                    elif BA_type == 'fixed_scale':
                        if idx < 30:
                            cost_function = ceres.CostFunction_PointsOnly(observations[i][j][0], observations[i][j][1], fx, fy, cx, cy, cam_pos[idx][0], cam_pos[idx][1], cam_pos[idx][2], \
                                                                        quat[idx][0], quat[idx][1], quat[idx][2], quat[idx][3])
                            problem.AddResidualBlock(cost_function, loss_function, point_coors[i])
                            flag = 1
                        else: # All other poses => Full BA
                            cost_function = ceres.CostFunction_FullBA(observations[i][j][0], observations[i][j][1], fx, fy, cx, cy)
                            problem.AddResidualBlock(cost_function, loss_function, quat[idx], cam_pos[idx], point_coors[i])
                            flag = 1
                        
                    else:
                        raise ValueError("Entered incorrect Bundle Adjustment type. Please choose between 'motion-only, 'point-only', 'complete' or 'sequential' .")

                    # Optionally set parameter bounds
                    if limit == 'Limited' and flag == 1:
                        if BA_type == 'complete' or BA_type == 'motion-only' or BA_type == 'fixed_scale':
                            for lim_idx in range(3): # Limit displacement of camera position
                                if abs(cam_pos[idx][lim_idx]) > 0.3:
                                    LL = cam_pos[idx][lim_idx] - abs(cam_pos[idx][lim_idx])
                                    UL = cam_pos[idx][lim_idx] + abs(cam_pos[idx][lim_idx])
                                else:
                                    LL = cam_pos[idx][lim_idx] - abs(cam_pos[idx][lim_idx]) - 0.3
                                    UL = cam_pos[idx][lim_idx] + abs(cam_pos[idx][lim_idx]) + 0.3
                                problem.SetParameterLowerBound(cam_pos[idx], lim_idx, LL) # set lower bound
                                problem.SetParameterUpperBound(cam_pos[idx], lim_idx, UL) # set upper bound
                            
                            for lim_idx in range(4): # Limit displacement of camera rotation
                                limits = sorted([0.5*quat[idx][lim_idx], 1.5*quat[idx][lim_idx]])
                                LL, UL = limits[0], limits[1]
                                if LL == UL:
                                    LL -= 0.001
                                    UL += 0.001
                                problem.SetParameterLowerBound(quat[idx], lim_idx, LL) # set lower bound
                                problem.SetParameterUpperBound(quat[idx], lim_idx, UL) # set upper bound   
                        
                        if BA_type == 'complete' or BA_type == 'point-only' or BA_type == 'fixed_scale':
                            for lim_idx in range(3): # Limit displacement of point coordinates
                                problem.SetParameterLowerBound(point_coors[i], lim_idx, point_coors[i][lim_idx]-1) # set lower bound of point
                                problem.SetParameterUpperBound(point_coors[i], lim_idx, point_coors[i][lim_idx]+1) # set upper bound of point
                        flag = 0
                    
        else:
            discarded_pts.append(i) 
    

    return problem, cnt_ret_pts, discarded_pts

# Run Ceres solver to perform the BA
def run_solver(problem, max_nbr_iter, verbose = 'output', report = 'full'):
    """
    Run the Ceres solver to optimize a given problem.

    Arguments:
    - problem: The ceres.Problem object representing the optimization problem.
    - max_nbr_iter (int): The maximum number of iterations for the solver.
    - verbose (str, optional): Verbosity of the solver. Valid options are: 'output' (default) to print progress to stdout, or 'silent' to suppress output.
    - report (str, optional): Level of reporting after solving. Valid options are: 'full' (default) to print a detailed report, or 'brief' to print a concise report.

    Returns:
    - summary: The ceres.Summary object containing the summary of the solver run.
    """

    # Define solver options
    options = ceres.SolverOptions()
    options.linear_solver_type = ceres.LinearSolverType.DENSE_SCHUR

    # Define verbosity of solver
    if verbose == 'output':
        options.minimizer_progress_to_stdout = True
    elif verbose == 'silent':
        options.minimizer_progress_to_stdout = False

    # Set max number of iterations
    options.max_num_iterations = max_nbr_iter

    # Run solver
    summary = ceres.Summary()
    ceres.Solve(options, problem, summary)
    if verbose == 'output':
        if report == "full":
            print(summary.FullReport())
        elif report == "brief":
            print(summary.BriefReport())
        else:
            raise ValueError("Invalid input. Please choose between 'full' or 'brief' report.")

    return summary

# Extract statistics from Ceres' solver summary
def solver_statistics(summary, verbose, cnt_ret_pts, points):
    """
    Compute statistics and provide information about the solver run.

    Arguments:
    - summary: The ceres.Summary object containing the summary of the solver run.
    - verbose (str): Verbosity level. Valid options are: 'output' to print information to stdout.
    - cnt_ret_pts (int): The count of retained points.
    - points: List of points used in the optimization.

    Returns:
    - cost_iter: List of costs for each iteration of the solver.
    - solution_stats: List containing solution statistics. It includes the following elements:
                      [nbr_iterations, cost_reduction, time].
    """
    nbr_iterations = len(summary.iterations)
    cost_iter = []
    for i in range(nbr_iterations):
        cost_iter.append(summary.iterations[i].cost)
    cost_reduction = cost_iter[-1]/cost_iter[0]*100
    if verbose == "output":
        print("Final cost = {:.2f}% of intitial cost".format(cost_reduction))
        print("{} out of {} points are retained.".format(cnt_ret_pts, len(points)))

    tot_time = summary.total_time_in_seconds
    min_time = summary.minimizer_time_in_seconds
    pre_time = summary.preprocessor_time_in_seconds
    post_time = summary.postprocessor_time_in_seconds
    time = np.array([tot_time, min_time, pre_time, post_time])

    solution_stats = []
    solution_stats.append(nbr_iterations), solution_stats.append(cost_reduction), solution_stats.append(time)

    return cost_iter, solution_stats


##################### Create and solve Bundle Adjustment problem for a single sequence (SS) #####################

def BundleAdjustment_SA(ORB_version, keyframes, points, T_BS, th_nbr_kf, cam_parameters, noise_type, noise_factor=0, threshold_noise = 20, BA_type='', max_nbr_iter=200, report='full', visualize=0, save=0, save_fig_path='', verbose='ouptut', limit= 'No limits', BA_algorithm = 'SABA', split_seq = 0):
    """
    Perform Bundle Adjustment (BA) with sensor-to-body transformation and noise addition.

    Arguments:
    - ORB_version: Version of the considered ORB-SLAM algorithm (not used).
    - keyframes: List of keyframes.
    - points: List of observed points.
    - T_BS: Sensor-to-body transformation matrix.
    - th_nbr_kf: Threshold number of keyframes for a point to be retained.
    - cam_parameters: Camera intrinsic parameters.
    - noise_type: Type of noise to be added.
    - noise_factor: Intensity of the noise (default = 0).
    - threshold_noise: Frame from which noise is applied (default = 20).
    - BA_type: Type of Bundle Adjustment to be performed. Valid options are: 'motion-only', 'point-only', 'complete', 'fixed_scale', 'sequential'.
    - max_nbr_iter: Maximum number of iterations for the solver (default = 200).
    - report: Level of report to be printed. Valid options are: 'full', 'brief' (default = 'full').
    - visualize: Flag indicating whether to visualize the cost function (default = 0).
    - save: Flag indicating whether to save the figures (default = 0).
    - save_fig_path: Path to save the figures (default = '').
    - verbose: Verbosity level. Valid options are: 'output' to print information to stdout or 'silent' to suppress output (default = 'ouptut').
    - limit (str): Parameter limit type. Valid options are: 'Limited' and 'No limits'.
    - BA_algorithm (str): Bundle adjustment algorithm used (options: 'MABA' or 'SABA').
    - split_seq: Length of the first sequence when MABA algorithm is considered (default = 0 for SABA).

    Returns:
    - org_data: Original data before any transformations or noise addition. It includes the following elements:
                [cam_pos_no_noise, point_coors_no_noise, cam_rot_vec_org].
    - noisy_data: Data after sensor-to-body transformation and noise addition. It includes the following elements:
                  [cam_pos_noise, point_coors_noise, rot_vec_noise_tf].
    - BA_data: Data after Bundle Adjustment. It includes the following elements:
               [cam_pos, point_coors, rot_vec_BA_tf].
    - cost_iter: List of costs for each iteration of the solver.
    - cnt_ret_pts: Count of retained points.
    - solution_stats: List containing solution statistics. It includes the following elements:
                      [nbr_iterations, cost_reduction, time].
    """
    # Extract data from provided keyframes and points
    point_coors_no_noise, _, frame_ids, observations = information_points(points)
    kf_identifier, _, cam_pos_no_noise, quat, cam_rot_mat = information_keyframes(keyframes)

    cam_rot_vec_org = copy.deepcopy(quat)

    # Convert keyframes and points to body frame using T_BS provided in sensor yaml file
    if np.any(T_BS):
        cam_pos_no_noise, quat, point_coors_no_noise = sensor2body(T_BS, cam_pos_no_noise, cam_rot_mat, point_coors_no_noise)
    
    # Add noise to cam position and points (will have no effect when noise_factor = 0)
    cam_pos_noise, point_coors_noise = add_noise_to_ORB_output(noise_type, BA_algorithm, points, cam_pos_no_noise, noise_factor, threshold_noise, split_seq) # generate noisy data
    cam_pos = copy.deepcopy(cam_pos_noise) # will be fed to BA
    point_coors =  copy.deepcopy(point_coors_noise)
    cam_rot_vec_transformed = copy.deepcopy(quat)


    input_quaternion = np.reshape([rotmat2quat(rot, "wxyz") for rot in cam_rot_vec_transformed], (-1,4))

    cam_intrinsics = np.array([458.654, 457.296, 367.215, 248.375])

    if BA_type == 'motion-only' or BA_type == 'point-only' or BA_type == 'complete' or BA_type == 'fixed_scale':

        # Create Ceres problem and add Residual Blocks
        problem, cnt_ret_pts, _ = residualblocks(frame_ids, kf_identifier, point_coors, observations, cam_pos, input_quaternion, cam_intrinsics, th_nbr_kf, BA_type, limit)

        # Run BA solver
        summary = run_solver(problem, max_nbr_iter, verbose, report)
        # Extract statistics
        cost_iter, solution_stats = solver_statistics(summary, verbose, cnt_ret_pts, points)


        # Optionally, visualize cost function
        if visualize:
            visualize_cost(cost_iter, save, save_fig_path)

    elif BA_type == 'sequential':
        smooth_threshold = 1.1 # Adapt for MH03

        # Perform 'motion-only' BA to refine camera poses
        problem_MO, cnt_ret_pts, _ = residualblocks(frame_ids, kf_identifier, point_coors, observations, cam_pos, input_quaternion, cam_intrinsics, th_nbr_kf, 'motion-only', limit) # Build NLLS problem
        summary_MO = run_solver(problem_MO, max_nbr_iter, verbose, report) # Run MO BA
        cam_pos_smooth = smoothen_trajectory(cam_pos, 0, smooth_threshold) # smoothen trajectory
        cost_iter, solution_stats = solver_statistics(summary_MO, verbose, cnt_ret_pts, points) # Extract statistics
        if visualize:
            visualize_cost(cost_iter, save, save_fig_path)
            compare_noisy_trajectories(cam_pos_no_noise, cam_pos_noise, cam_pos, cam_pos_smooth, 0, 0, 'Motion-Only', save, save_fig_path)

        # Perform 'point-only' BA to refine point coor => at this stage, the cam_pos and input_quaternion at the input of 'residualblocks' are the refined ones by MO
        problem_PO, cnt_ret_pts, _ = residualblocks(frame_ids, kf_identifier, point_coors, observations, cam_pos, input_quaternion, cam_intrinsics, th_nbr_kf, 'point-only', limit, 30) # Build NLLS problem
        summary_PO = run_solver(problem_PO, max_nbr_iter, verbose, report) # Run MO BA
        cost_iter, solution_stats = solver_statistics(summary_PO, verbose, cnt_ret_pts, points) # Extract statistics
        if visualize:
            visualize_cost(cost_iter, save, save_fig_path)
            plot_noisy_pointclouds(point_coors_no_noise, point_coors_noise, point_coors, 'Point-Only', save, save_fig_path)
        

        # Perform 'complete' BA to jointly refine both => at this stage, the point_coors are refined by the PO as well
        problem_seq, _, _ = residualblocks(frame_ids, kf_identifier, point_coors, observations, cam_pos_smooth, input_quaternion, cam_intrinsics, th_nbr_kf, 'complete', limit) # Build NLLS problem
        summary_seq = run_solver(problem_seq, max_nbr_iter, verbose, report) # Run complete BA
        cost_iter, solution_stats = solver_statistics(summary_seq, verbose, cnt_ret_pts, points) # Extract statistics
        if visualize:
            visualize_cost(cost_iter, save, save_fig_path)

    else:
        raise ValueError("Entered incorrect Bundle Adjustment type. Please choose between 'motion-only, 'points-only', 'complete' or 'sequential' .")
    
    # Transform camera rotations to coordinate frame of ground truth
    T_BS = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                     [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                     [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                     [0.0, 0.0, 0.0, 1.0]])
    
    R_BS, t_BS = T_BS[:3,:3], T_BS[:3, 3]
    T_BS = np.reshape(inverse_transformation_matrix(R_BS, t_BS, "vector"), (4,4))
 
    # T_BS = np.eye(4)
    rot_mat_noise = [np.reshape(rot, (3,3)) for rot in cam_rot_vec_transformed]
    cam_rot_after_BA = [np.reshape(quat2rotmat(quat, "vector", "wxyz"), (3,3)) for quat in input_quaternion]

    _, rot_vec_noise_tf, _ = sensor2body(T_BS, cam_pos_no_noise, rot_mat_noise, point_coors_no_noise)
    _, rot_vec_BA_tf, _ = sensor2body(T_BS, cam_pos_no_noise, cam_rot_after_BA, point_coors_no_noise)


    # Summarize data in one data struct
    org_data, noisy_data, BA_data = [], [], []
    org_data.append(cam_pos_no_noise), org_data.append(point_coors_no_noise), org_data.append(cam_rot_vec_org) # cam rotation before translation
    noisy_data.append(cam_pos_noise), noisy_data.append(point_coors_noise), noisy_data.append(rot_vec_noise_tf) # data after transformation but before BA
    BA_data.append(cam_pos), BA_data.append(point_coors), BA_data.append(rot_vec_BA_tf) # data after BA


    return org_data, noisy_data, BA_data, cost_iter, cnt_ret_pts, solution_stats

