################################ Import modules and functions ################################
import os
import sys
lib_location = "/home/pb/Documents/Thesis/Scripts/lib"
PyCeres_location = "/home/pb/ceres-bin/lib"
sys.path.insert(0, lib_location)
sys.path.insert(0,PyCeres_location)
import PyCeres as ceres # Import the Python Bindings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2
import multiprocessing as mp
import time
import copy

from matplotlib import image
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


from functions import parsedata, information_keyframes, information_points, extract_point
from bundle_adjustment import visualize_cost
from functions import float2int
from rotation_transformations import quat2rotmat, rotmat2quat
from project_keyframe import find_imgpath

################################       Define functions       ################################

## Transformations
def scale_adaptive_icp(source, target, max_iterations=50, tolerance=1e-5): # Perform scale-adaptive ICP
    """
    Performs scale-adaptive ICP between two point clouds.

    Arguments:
    - source -- the source point cloud
    - target -- the target point cloud
    - max_iterations -- the maximum number of ICP iterations (default 50)
    - tolerance -- the convergence tolerance (default 1e-5)

    Returns:
    - R -- the rotation matrix
    - t -- the translation vector
    - s -- the scale factor
    """
    # Initialize the transformation parameters to the identity matrix
    R = np.identity(3)
    t = np.zeros(3)
    s = 1

    # Start the ICP iteration loop
    for i in range(max_iterations):
        # Transform the source point cloud using the current transformation
        transformed = s * np.dot(R, source.T).T + t

        # Find the nearest neighbors between the transformed source and target point clouds
        # distances, indices = nn_matching(transformed, target)
        distances = dist2match(transformed, target)

        # Compute the weighted mean of the nearest neighbor correspondences
        mean_source = np.mean(source, axis=0)
        # mean_target = np.mean(target[indices], axis=0)
        # cov = np.dot((source - mean_source).T, target[indices] - mean_target)
        mean_target = np.mean(target, axis=0)
        cov = np.dot((source - mean_source).T, target - mean_target)
        U, S, V = np.linalg.svd(cov)
        Rn = np.dot(V.T, U.T)
        if np.linalg.det(Rn) < 0:
            V[2,:] *= -1 # ensure Rn represents a proper rotation matrix
            Rn = np.dot(V.T, U.T)
        tn = mean_target - s * np.dot(Rn, mean_source)

        # Compute the residual error and check for convergence
        error = np.sum(distances)
        if abs(error) < tolerance:
            print("Convergence reached after {} iterations.".format(i))
            break

        # Update the transformation parameters
        R = np.dot(Rn, R)
        t = s * np.dot(Rn, t) + tn
        s *= np.exp(np.log(S[2] / S[0]) / 3.0)

        print("R:", R)
        print("t:", t)
        print("s:", s)

    return R, t, s

def scaled_PCR(source, target, max_iterations=50, plot_cost=0, verbose=1): # Perform Scaled Point Cloud Registration
    """
    This function performs scale-adaptive ICP between two point clouds. To this end, a NLLS-optimization problem is solved using the Ceres solver.

    Arguments:
    - source -- Source point cloud (point cloud that will be transformed)
    - target -- Target point cloud (to which the source point cloud will be mapped)
    - max_iterations -- Maximum number of solver iterations (default = 50)
    - plot_cost -- Boolean: toggle plot of cost function ono or off (default = 0)
    - verbose -- Boolean; 1 to print results of NLLS optimization, 0 to supress report (default = 1)


    Returns:
    - R -- 3x3 rotation matrix
    - t -- 3x1 translation vector
    - s -- 3x1 scale factor
    - cost_iter -- Array containg the total cost after every iteration of the optimization process.
    
    """
    # Create PyCeres problem
    problem = ceres.Problem()
    loss_function = ceres.HuberLoss(0.1) 

    # Initialize transformation parameters
    s = np.array([1.0, 1.0, 1.0])
    t = np.array([0.0, 0.0, 0.0])
    R = np.eye(3).flatten()

    # Create Residual Blocks for optimization problem
    for i in range(len(source)): # loop over all the points in the provided pointcloud
        source_coor = source[i]
        target_coor = target[i]
        
        # Define cost function for the point
        cost_function = ceres.PCR_CostFunction(target_coor[0], target_coor[1], target_coor[2], source_coor[0], source_coor[1], source_coor[2])
        problem.AddResidualBlock(cost_function, loss_function, s, R, t) 
    # Specify solver options
    options = ceres.SolverOptions()
    options.linear_solver_type = ceres.LinearSolverType.DENSE_SCHUR
    if verbose:
        options.minimizer_progress_to_stdout = True
    else:
        options.minimizer_progress_to_stdout = False
    options.max_num_iterations = max_iterations

    summary = ceres.Summary()
    ceres.Solve(options, problem, summary)
    if verbose:
        print(summary.FullReport())

    # Extract statistics
    nbr_iterations = len(summary.iterations)
    cost_iter = []
    for i in range(nbr_iterations):
        cost_iter.append(summary.iterations[i].cost)
    cost_reduction = cost_iter[-1]/cost_iter[0]*100
    if verbose:
        print("Final cost = {:.2f}% of intitial cost".format(cost_reduction))

    if plot_cost:
        visualize_cost(cost_iter, 0)
    

    return R, t, s, cost_iter

def scaled_PCR(source, target, max_iterations=50, plot_cost=0, verbose=1):
    """
    This function performs scale-adaptive ICP between two point clouds. To this end, a NLLS-optimization problem is solved using the Ceres solver.

    Arguments:
    - source -- Source point cloud (point cloud that will be transformed)
    - target -- Target point cloud (to which the source point cloud will be mapped)
    - max_iterations -- Maximum number of solver iterations (default = 50)
    - plot_cost -- Boolean: toggle plot of cost function ono or off (default = 0)
    - verbose -- Boolean; 1 to print results of NLLS optimization, 0 to supress report (default = 1)


    Returns:
    - R -- 3x3 rotation matrix
    - t -- 3x1 translation vector
    - s -- 3x1 scale factor
    - cost_iter -- Array containg the total cost after every iteration of the optimization process.
    
    """
    # Create PyCeres problem
    problem = ceres.Problem()
    loss_function = ceres.HuberLoss(0.1) 

    # Initialize transformation parameters
    s = np.array([1.0, 1.0, 1.0])
    t = np.array([0.0, 0.0, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])  # initialize the rotation as a quaternion
    # print("Transformation parameters initialized")

    # Create Residual Blocks for optimization problem
    for i in range(len(source)): # loop over all the points in the provided pointcloud
        source_coor = source[i]
        target_coor = target[i]
        # Define cost function
        # print("source coordinates:", source_coor)
        # print("target coordinates:", target_coor)
        cost_function = ceres.PCR_CostFunction(target_coor[0], target_coor[1], target_coor[2], source_coor[0], source_coor[1], source_coor[2])
        problem.AddResidualBlock(cost_function, loss_function, s, q, t) 
        # print("Residual block added")
        # problem.SetParameterization(q,  ceres.QuaternionParameterization())
        # problem.AddParameterBlock(q, 4, QuaternionManifold())
        # print("Parameterization set")

    # Specify solver options
    options = ceres.SolverOptions()
    # options.jacobian_scaling = ceres.QuaternionNormalization()
    options.linear_solver_type = ceres.LinearSolverType.DENSE_SCHUR
    if verbose:
        options.minimizer_progress_to_stdout = True
    else:
        options.minimizer_progress_to_stdout = False
    options.max_num_iterations = max_iterations

    summary = ceres.Summary()
    print("Run solver")
    ceres.Solve(options, problem, summary)
    if verbose:
        print(summary.FullReport())

    # Extract statistics
    nbr_iterations = len(summary.iterations)
    cost_iter = []
    for i in range(nbr_iterations):
        cost_iter.append(summary.iterations[i].cost)
    cost_reduction = cost_iter[-1]/cost_iter[0]*100
    if verbose:
        print("Final cost = {:.2f}% of intitial cost".format(cost_reduction))

    if plot_cost:
        visualize_cost(cost_iter, 0)

    # Convert the quaternion to a rotation matrix
    # print("quaternion wxyz=", q)
    print(type(q))
    q = np.append(q[1:], q[0])
    # print("quaternion xyzw=", q)
    R = quat2rotmat(q, "vector")
    # print(np.reshape(R, (3,3)))
    R = np.array([R[:3], R[3:6], R[6:]])
    # print("Final R:", R)

    return R, t, s, cost_iter

def rigid_transform(source, R: np.ndarray, t, s, mode): # Apply transformations to point / keyframe
    if R.shape != (3,3):
        R = np.array([R[0][:3], R[0][3:6], R[0][6:]]) 
    
    s = np.diag(np.append(s,1))
    t = np.reshape(t,(-1,1))
    T = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
    if mode == 'point':
        src = source[0]
        res =  [np.dot(np.dot(s,T), np.append(src,1).T)[:-1] for src in source]

    elif mode == 'keyframe':
        res = [[], []]
        for P in source:
            transform = np.dot(T, P)
            rot = transform[:3,:3]
            translation = transform[:3,-1]
            translation_scaled = np.dot(s[:3, :3], translation)
            res[0].append(transform[:3,:3])
            res[1].append(translation_scaled)

    return res

def transform_sequences(point_coor, cam_pos, cam_rot, R, t, s):
    """
    Use obtained R, t and s to transform data (points + keyframe poses) from sequence 2 such that they correspond to sequence 1.
    """
    # Transform point coordinates
    coor_tf = rigid_transform(point_coor, R, t, s, 'point')

    # Transform keyframes
    P_seq = [np.vstack((np.hstack((np.reshape(quat2rotmat(rot, "vector"), (3,3)), np.reshape(pos,(-1,1)))), [0, 0, 0, 1])) for rot, pos in zip(cam_rot, cam_pos)]
    kf_seq2_tf = rigid_transform(P_seq, R, t, s, 'keyframe')
    cam_rot_tf, cam_pos_tf = kf_seq2_tf[0], kf_seq2_tf[1]

    return coor_tf, cam_rot_tf, cam_pos_tf

## Visualizations
def visualize_pointcloud(source, source_tf, target, links=0, fig_title=""): # Plot pointclouds
    # Initialize figure
    fig = plt.figure(figsize=(32,18))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('\n\nx position',fontsize = 30, fontweight ='bold')
    ax.set_ylabel('\n\ny position',fontsize = 30, fontweight ='bold')
    ax.set_zlabel('\n\nz position',fontsize = 30, fontweight ='bold')
    ax.set_title(fig_title, fontsize = 50, fontweight = 'bold', y = 1.1)
    ax.tick_params(axis='both', which='major', labelsize=24)

    # Plot point clouds before and after ICP
    ax.scatter([pt[0] for pt in target], [pt[1] for pt in target],[pt[2] for pt in target], marker='o',c='m', label = 'Target points', s = 50)
    ax.scatter([pt[0] for pt in source], [pt[1] for pt in source],[pt[2] for pt in source], marker='x',c='r', label = 'Source points before PCR', s = 150)
    ax.scatter([pt[0] for pt in source_tf], [pt[1] for pt in source_tf],[pt[2] for pt in source_tf], marker='d',c='g', label = 'Source points after PCR', s = 150)

    if links:
        for i in range(len(source)):
            ax.plot([source_tf[i][0], source[i][0]], [source_tf[i][1], source[i][1]], [source_tf[i][2], source[i][2]], c='k', linestyle=':', linewidth=1)
            ax.plot([source_tf[i][0], target[i][0]], [source_tf[i][1], target[i][1]], [source_tf[i][2], target[i][2]], c='r', linestyle=':', linewidth=2)

    ax.legend(fontsize = 24, markerscale = 2)
    plt.show()

def visualize_matches(img_path, obs): # Plot keypoints in specific frames to show correpondencies
    fig, axs = plt.subplots(1,2,figsize=(32,18))
    fig.tight_layout()
    

    for i in range(len(obs)):
        for j in range(len(obs[i])):
            x_obs, y_obs = obs[i][j][0], obs[i][j][1]

            if j == 0:
                axs[i].scatter(x_obs, y_obs, linewidth=4, label= 'Keypoints' )
            else:
                axs[i].scatter(x_obs, y_obs, linewidth=4)                         

            # Add written number to point
            axs[i].text(x_obs+10, y_obs+10, str(j+1), fontsize=12, fontweight = "bold", color='red', 
                        horizontalalignment='center', verticalalignment='center')
            
            # ax.plot([x_proj, x_obs], [y_proj, y_obs], "r:", linewidth=3)

    
    # fig.legend(loc = "lower center",fontsize=18)
    fig.suptitle("Matching keypoints", fontsize = 40, fontweight = "bold", y= 0.85)
    for i, ax in enumerate(axs):
        ax.set_xlim([-50, 800]), ax.set_ylim([570, -50])
        # ax.set_xlim([0, 752]), ax.set_ylim([480, 0])
        ax.imshow(image.imread(img_path[i]))

    plt.show()

def visualize_matches_lines(img_path, obs):
    # Load the left and right images
    left_image = cv2.imread(img_path[0])
    right_image = cv2.imread(img_path[1])
    width, height, _ = left_image.shape

    # Combine the left and right images into a single image
    combined_image = np.concatenate((left_image, right_image), axis=1)

    # Generate a list of 48 line colors
    line_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
                   (0, 128, 128), (128, 0, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (0, 64, 64), (64, 0, 64), (192, 0, 0), (0, 192, 0),
                   (0, 0, 192), (192, 192, 0), (0, 192, 192), (192, 0, 192), (128, 64, 0), (64, 128, 0), (64, 0, 128), (128, 128, 64), (64, 128, 128), (128, 64, 128),
                   (255, 128, 0), (128, 255, 0), (128, 0, 255), (255, 255, 128), (128, 255, 255), (255, 128, 255), (192, 64, 0), (64, 192, 0), (64, 0, 192), (192, 192, 64),
                   (64, 192, 192), (192, 64, 192), (255, 192, 0), (192, 255, 0), (192, 0, 255), (255, 255, 192), (192, 255, 255), (255, 192, 255)]

    for i in range(len(obs[0])):
        # Define the coordinates of the pixels to draw the line between
        obs_left, obs_right = (int(np.round(obs[0][i][0])), int(np.round(obs[0][i][1]))), (int(np.round(obs[1][i][0]))+height,  int(np.round(obs[1][i][1])))


        # Draw the line between the pixels
        line_color = line_colors[i % len(line_colors)]
        line_thickness = 2
        cv2.line(combined_image, obs_left, obs_right, line_color, line_thickness, cv2.LINE_AA)
        cv2.circle(combined_image, obs_left, 5, line_color, -1)
        cv2.circle(combined_image, obs_right, 5, line_color, -1)
        
        # cv2.circle(combined_image, (right_pixel_x[i] + left_image.shape[1], right_pixel_y[i]), marker_size, marker_color, -1, cv2.LINE_AA)




    # Show the combined image with the line drawn between the pixels
    cv2.imshow('Combined Image', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_observation(img_path, obs, text, kf_id, save=0, path=''):
    fig = plt.figure(figsize=(32,18))
    for i in range(len(obs)):
        x_obs, y_obs = obs[i][0], obs[i][1]

        if i == 0:
            plt.scatter(x_obs, y_obs, linewidth=4, label= 'Keypoints', c='r')
        else:
            plt.scatter(x_obs, y_obs, linewidth=4, c='r')                         

        # Add written number to point
        if text:
            plt.text(x_obs+10, y_obs+10, str(i+1), fontsize=12, fontweight = "bold", color='red', 
                        horizontalalignment='center', verticalalignment='center')
        
    
    # fig.legend(loc = "lower center",fontsize=18)
    plt.title("Observed keypoints", fontsize = 40, fontweight = "bold", y= 1)
    plt.xlim([-50, 800]), plt.ylim([570, -50])
    # plt.xlim([0, 752]), plt.ylim([480, 0])
    # plt.tick_params('both', labelleft = False, labelbottom = False)
    plt.imshow(image.imread(img_path))

    if save:
        plt.savefig(path + 'kf{}'.format(kf_id) + '.png', bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
        plt.close('all')
    else:
        plt.show()

def visualize_keypoints(directory, keyframes, points, text, kf_id_range, save=0, path=''):
    for kf_id in kf_id_range:
        _, _, _, obsv, _ = extract_point(points, kf_id)
        _, obsvDis = [obs[:2] for obs in obsv], [obs[2:] for obs in obsv]
        idx = [keyframe[0].astype(int) for keyframe in keyframes].index(kf_id)
        kf = keyframes[idx]
        timestamp = kf[1]
        img_path = find_imgpath(directory,  timestamp)
        visualize_observation(img_path, obsvDis, text, kf_id, save, path)

def plot_trajectories_combined(traj_org, limit, sequences):

    # Initialize figures
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(32, 18)
    fig.subplots_adjust(top=0.96, bottom=0.06, left=0.06, right=0.99, hspace=0.45, wspace=0.2)

    # Initialize variables & plot
    x_org = [camera_translation[0] for camera_translation in traj_org]
    y_org = [camera_translation[1] for camera_translation in traj_org]
    z_org = [camera_translation[2] for camera_translation in traj_org]
    
    ax[0].scatter(range(len(x_org[:limit])), x_org[:limit], linewidths = 3, marker = "o", color = "blue", s = 50,
                  label = sequences[0])
    ax[1].scatter(range(len(y_org[:limit])), y_org[:limit], linewidths = 3, marker = "o", color = "blue", s = 50,
                  label = sequences[0])
    ax[2].scatter(range(len(z_org[:limit])), z_org[:limit], linewidths = 3, marker = "o", color = "blue", s = 50,
                  label = sequences[0])
    
    ax[0].scatter(range(limit, len(x_org)), x_org[limit:], linewidths = 3, marker = "o", color = "red", s = 50,
                  label = sequences[1])
    ax[1].scatter(range(limit, len(y_org)), y_org[limit:], linewidths = 3, marker = "o", color = "red", s = 50,
                  label = sequences[1])
    ax[2].scatter(range(limit, len(z_org)), z_org[limit:], linewidths = 3, marker = "o", color = "red", s = 50,
                  label = sequences[1])
            
    ax[0].set_title('X position', fontsize = 40, fontweight ='bold')
    ax[1].set_title('Y position', fontsize = 40, fontweight ='bold')
    ax[2].set_title('Z position', fontsize = 40, fontweight ='bold') 

    for axis in ax:
        axis.set_xlabel('Keyframe', fontsize = 30), axis.set_ylabel('Position', fontsize = 30)
        axis.legend(fontsize = 24, markerscale=2)
        axis.minorticks_on(), axis.ticklabel_format(useOffset=False)
        axis.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
        # axis.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')
        axis.set_xlim([0, len(traj_org)+75])
        axis.tick_params(axis='both', which='major', labelsize=24)

    plt.show()

def visualize_matching_observations(img_path: str, obs, kf_id, save_fig, save_path =''):

    data= image.imread(img_path)

    fig, ax = plt.subplots(figsize=(16,9))
    for i in range(len(obs)):
        x_obs, y_obs = obs[i][0], obs[i][1]
        ax.scatter(x_obs, y_obs, color="blue", linewidth=4)

        ax.text(x_obs+10, y_obs+10, str(obs[i][2]), fontsize=12, fontweight = "bold", color='red', 
            horizontalalignment='center', verticalalignment='center')

    ax.set_title("Keyframe {}".format(int(kf_id)), fontsize = 22, fontweight = "bold")
    ax.set_xlim([-20, 775]), ax.set_ylim([500, -20])
    ax.imshow(data)

    
    if save_fig == 0:
        plt.show()
    else:
        # save_path = "/home/pb/Documents/Thesis/Figures/ORB SLAM/Reprojection KeyPoints/{}/{}/kf_{}.png".format(ORB_version, sequence, int(kf_id))
        plt.savefig(save_path + "kf_{}.png".format(int(kf_id)), bbox_inches = 'tight', pad_inches = 0.2, transparent = False)
        plt.close('all')

## Matching points across sequences

# Metric: Hamming distance
def count_set_bits(xor, LOT): # Count number of 1's in binary representation of element
    """
    Computes the number of set bits for each element in the given array xor.
    """
    count = np.zeros((xor.shape[0], xor.shape[1]))
    for i in range(xor.shape[0]):
        for j in range(xor.shape[1]):
            arr = xor[i][j]
            count[i,j] = sum([LOT[x] for x in arr])

    return count

def compute_hamming_distance(args): # Compute the hamming distance between elements of arrays
    """
    Computes the Hamming distance between two arrays using a look-up table.
    This function is intended to be called by a worker process.
    """
    arr1, arr2, LOT_256 = args
    xor = np.bitwise_xor(arr1[:, np.newaxis], arr2)
    # xor = np.bitwise_xor.outer(arr1, arr2)
    return count_set_bits(xor, LOT_256)

def parallel_hamming_distance(arr1, arr2): # Parallelized version of hamming distance calculation code
    """
    Computes the Hamming distance between every pair of values in arr1 and arr2 using multiple processes.

    Parameters:
    - arr1 (numpy.ndarray) -- Array of shape (n, 32) containing integer values between 0 and 255.
    - arr2 (numpy.ndarray) --  Array of shape (m, 32) containing integer values between 0 and 255.

    Returns:
     - numpy.ndarray: Array of shape (n, m) containing the Hamming distance between every pair of values in arr1 and arr2.
    """
    # Create a pool of worker processes
    pool = mp.Pool()

    # Create a look-up table to count number of set bits
    LOT_256 = bytes(bin(x).count("1") for x in range(256))

    # Split the task into smaller chunks
    chunks = [(arr1[i:i+100], arr2, LOT_256) for i in range(0, arr1.shape[0], 100)]

    # Compute the Hamming distance for each chunk in parallel
    results = pool.map(compute_hamming_distance, chunks)

    # Merge the results from all chunks
    hamming_dist = np.concatenate(results, axis=0)

    # Close the pool of worker processes
    pool.close()

    return hamming_dist

def hamming_distance(arr1, arr2): # Sequential hamming distance calculation code
    """
    Computes the Hamming distance between every pair of values in arr1 and arr2.

    Parameters:
    - arr1 (numpy.ndarray) -- Array of shape (n, 32) containing integer values between 0 and 255.
    - arr2 (numpy.ndarray) --  Array of shape (m, 32) containing integer values between 0 and 255.

    Returns:
     - numpy.ndarray: Array of shape (n, m) containing the Hamming distance between every pair of values in arr1 and arr2.
    """
    # Compute the bitwise XOR between every pair of values
    xor = np.bitwise_xor(arr1[:, np.newaxis], arr2)
    # print("\n",xor.shape)

    # Count the number of set bits in the resulting XOR
    LOT_256 = bytes(bin(x).count("1") for x in range(256)) # look-up table to count number of set bits
    hamming_dist = count_set_bits(xor, LOT_256)

    return hamming_dist

# Matching     
def descriptors_distance(descriptors1, descriptors2, type_norm, parallel): # Calculate distance between desciptors of two sets / sequences
    """
    This function performs feature matching based on minimizing the distance between the descriptors of every feature. The user can choose between the 
    Euclidean (L2) norm or the Hamming distance when binary descriptors (such as ORB) are considered. These distances are computes based on an efficient
    matrix multiplication, which results in a much faster computation than for nested for-loops.
    
    Arguments:
    - descriptors1 -- Descriptors of all n points observed by the first sequence (n x l), with l = length descriptor
    - descriptors2 -- Descriptors of all m points observed by the second sequence (m x l)
    - type_norm -- Either 'L2' or 'hamming', depending on the choice of distance metric

    Returns:
    - distances -- n x m matrix containing the respective distances, according to the chosen norm 
    """      
    if type_norm == "L2":
    # Transform descriptors into 'N x 32' matrices
        F1 = np.reshape(descriptors1, (-1,32))
        F2 = np.reshape(descriptors2, (-1,32))

        # Create temporary matrices
        B = 2*np.matmul(F1, F2.T)
        F1_sq = np.reshape(np.sum(np.multiply(F1,F1), axis = 1), (-1,1))
        F2_sq = np.reshape(np.sum(np.multiply(F2,F2), axis = 1), (1,-1))

        A = F1_sq + F2_sq

        distances = np.sqrt(A-B)

    elif type_norm == "hamming":
        if parallel:
            print(type(descriptors1))
            distances = parallel_hamming_distance(descriptors1, descriptors2)
        else:
            distances = hamming_distance(descriptors1, descriptors2)

    return distances

def descriptor_matching(dist, thresh = 0.5, verbose = 0): # Match descriptors by finding minimal distance    
    """
    This function performs feature matching based on the Euclidean or Hamming norm of the desciptors.
    If the Euclidean norm is considered, Lowe's ratio test is implemented to guarantee an unanmbiguous feature matching and to reduce the effect of incorrect
    feature associations.
    
    Arguments:
    - dist -- n x m matrix containing the respective distances between the descriptors, generated by the 'descriptors_distance' function
    - thresh -- Maximal acceptance threshold for the ratio test (default = 0.5)

    Returns:
    - match_idx -- Array containing for every point in sequence 1 the index of a matching point in sequence 2. 
                   If no unambiguous mapping can be established, the idx for this point will be NaN.
    - min_dist -- Array containing the two smallest distances for every point of sequence 1
    - ratio_min -- Result of division of two smallest distances. If ratio_min < threshold, the matching is accepted.
    """     

    # Find the two smallest elements in each row & idx of smallest
    min_dist = np.partition(dist, 2, axis=1)[:, :2]
    min_idx = np.argmin(dist, axis=1)

    # Find the ratio of the two minimal distances
    ratio_min = []
    for i in range(dist.shape[0]):
        ratio_min.append(min_dist[i][0]/min_dist[i][1])
    
    # If ratio < threshold, accept association
    match_idx = []
    for i in range(len(ratio_min)):
        if ratio_min[i] <= thresh:
            match_idx.append(min_idx[i])
        else:
            match_idx.append(np.nan)

    if verbose:
        print(' i ','pt', '  min dist   ', 'ratio')
        print('-------------------------------------------------------')

        for i in range(len(match_idx)):
            if not np.isnan(match_idx[i]):
                print(i+1, match_idx[i], min_dist[i], ratio_min[i])
        print('-------------------------------------------------------')
        print("")
    return match_idx, min_dist, ratio_min

def point_matching(match_idx, obsv1, obsv2, coor1, coor2):# Create pairs of matching points and observations
    """
    This function will create pairs of matching observations and point coordinates, based on the 'match_idx' established with the help of a descriptor matching.
    """
    obsv1_match, obsv2_match = [], []
    coor1_match, coor2_match = [], []
    for i in range(len(match_idx)):
        if not np.isnan(match_idx[i]): # ratio test passed => match accepted
            obsv1_match.append(obsv1[i])
            obsv2_match.append(obsv2[match_idx[i]])

            coor1_match.append(coor1[i])
            coor2_match.append(coor2[match_idx[i]])
    
    # print(obsv1_match)
    # obsv1_match = np.reshape(obsv1_match, (-1,2))
    # obsv2_match = np.reshape(obsv2_match, (-1,2))
  
    return obsv1_match, obsv2_match, coor1_match, coor2_match

def match_PointsInKf(keyframes, points, kf_id, visualize = 0, dir = [], verbose=0, parallel=0): # Match points in specific keyframes from different sequences
    """
    Match keypoints detected in specific frames to estimate the pose transformation between two sequences.
    First, the detected keypoints are matched based on their descriptor. Secondly, these matched pairs are fed to a Point Cloud Registration (PCR)
    algorithm to find the relative pose transformation between the different sequences.

    Arguments:
    - keyframes -- Keyframes for both sequences
    - points -- All points observed in both sequences
    - kf_id -- The keyframe identifier of the considered keyframes for both sequences, provides as list ([kf_id1, kf_id2])
    - visualize -- Boolean, indicate whether plots of the keyframes and the PCR are shown (default = 0)
    - dir -- Contains paths were camera images of both sequences are stored ([dir1, dir2], default = []\n
             E.g dir = ['./EUROC dataset/ASL/MH_01_easy/mav0/cam0/data/', './EUROC dataset/ASL/MH_02_easy/mav0/cam0/data/']
    - verbose -- Boolean, indicate whether output report of Ceres should be printed (default = 0)

    
    Returns:
    - R -- 3x3 rotation matrix
    - t -- 1x3 translation vector
    - s -- 1x3 scale factor\n
    These parameters can be used to match the points and camera poses of sequence 2 to those of sequence 1.
    """                     

    # Extract relevant information from provided keyframes
    keyframes_seq1, keyframes_seq2 = keyframes[0], keyframes[1]
    idx1 = [keyframe[0].astype(int) for keyframe in keyframes[0]].index(kf_id[0])
    idx2 = [keyframe[0].astype(int) for keyframe in keyframes[1]].index(kf_id[1])

    kf1 = keyframes_seq1[idx1]
    kf2 = keyframes_seq2[idx2]

    # Find all points seen in the selected keyframes + their cooridinates, descriptors and observations
    coor1, descr1, frame_ids1, obsv1, points_kf1 = extract_point(points[0], kf_id[0])
    coor2, descr2, frame_ids2, obsv2, points_kf2 = extract_point(points[1], kf_id[1])

    # all points in all frames of sequence 2
    point_coor, descr, frame_ids, observations = information_points(points[1])

    # Perform descriptor matching on these points
    # print(type(descr1))
    dist = descriptors_distance(descr1, descr2, 'hamming', parallel)
    # np.set_printoptions(threshold=np.inf)
    # print(dist)
    match_idx, min_dist, ratio_min = descriptor_matching(dist, 0.5, 1)
    obsv1_match, obsv2_match, coor1_match, coor2_match = point_matching(match_idx, obsv1, obsv2, coor1, coor2)
    obsv1_match_Dis, obsv2_match_Dis = [[obs[-2], obs[-1]] for obs in obsv1_match], [[obs[-2], obs[-1]] for obs in obsv2_match]
    # print(len(obsv1_match))

    # dist_seq = descriptors_distance(descr1, descr, 'hamming', parallel)
    # match_idx_seq, min_dist_seq, ratio_min_seq = descriptor_matching(dist_seq, 0.8, 1)
    
    if len(obsv1_match) == 0:
        print("No matching keypoints in the considered keyframes, aborting now.")
        return 0, 0, 0
    
    # Run PCR to calculate relative pose transformation (map points of seq 2 (src) to points of seq 1 (target))
    R, t, s, _ = scaled_PCR(source = coor2_match, target = coor1_match, max_iterations=50, plot_cost=visualize, verbose=verbose)
    coor2_match_tf = rigid_transform(coor2_match, R, t, s, 'point') 

    if visualize:
        if len(dir) != 0:
            timestamp1, timestamp2 = kf1[1],kf2[1]
            img_path = [find_imgpath(dir[0], timestamp1), find_imgpath(dir[1], timestamp2)]

            visualize_matches_lines(img_path, [obsv1_match_Dis, obsv2_match_Dis])
        visualize_pointcloud(coor2_match, coor2_match_tf, coor1_match, 1, "Scaled Point Cloud Registration")

    return R, t, s

def geometric_verification(match_idx, point_coor_tar, point_coor_src, verbose): # Check if matched points are close in space
    matches = match_idx
    print("number of matches before verification:", np.sum(~np.isnan(matches)))
    for i in range(len(match_idx)):
        if not np.isnan(match_idx[i]):
            coor_tar = point_coor_tar[i]
            coor_src = point_coor_src[match_idx[i]]

            L2 = np.linalg.norm(coor_tar - coor_src)
            L2_r = np.linalg.norm(coor_tar - coor_src)/np.linalg.norm(coor_tar)

            if L2_r > 0.10: # 5% deviation in estimated point position
                matches[i] = np.nan
                if verbose:
                    print("Match {} discarded because point coordinates are too far apart.".format(i+1))

            if verbose:
                print("Tar coor of pair {}: ".format(i+1), coor_tar)
                print("Src coor of pair {}: ".format(i+1), coor_src)
                print("L2 norm of pair {}: ".format(i+1), np.linalg.norm(coor_tar - coor_src))
                print("L2 norm ratio of pair {}: ".format(i+1), L2_r)
                print("")
    
    return matches

def keypoint_matching(points, visualize, verbose, r = 0.8): # Match keypoints from two sequences based on a descriptor matching and a geometric verification
    """
    First sequence: target
    Second sequence: source
    """
    # Extract point information for both sequences
    point_coor_tar, descr_tar, _, obsv_tar = information_points(points[0])
    point_coor_src, descr_src, _, obsv_src = information_points(points[1])

    # Perform  descriptor matching
    print("start dist calculation")
    dist = parallel_hamming_distance(descr_tar, descr_src)
    print("end distance calc")
    match_idx, _, _ = descriptor_matching(dist, 0.3, verbose) # Strict ratio test to ensure correct matches for transformations
    # np.save('dist_MH01_MH02.npy', dist)
    # print("dist saved")
    _, _, coor_tar_match, coor_src_match = point_matching(match_idx, obsv_tar, obsv_src, point_coor_tar, point_coor_src)

    print("Number of matches for SPCR:", len(coor_tar_match))

    # Transform point clouds to common reference frame
    R, t, s, _ = scaled_PCR(source = coor_src_match, target = coor_tar_match, max_iterations=50, plot_cost=visualize, verbose=verbose)
    transformation_parameters = [R, t, s]
    print("transformation parameters:", transformation_parameters)

    # Geometric verification: check if transformed source coordinates are indeed close to target points
    coor_src_match_tf = rigid_transform(coor_src_match, R, t, s, 'point')
    norm = []
    for i in range(len(coor_tar_match)):
        norm_i = np.linalg.norm(coor_tar_match[i] - coor_src_match_tf[i])
        print("match {}; coordinates target = {}, coordinates source transform = {}, norm = {}".format(i, coor_tar_match[i], coor_src_match_tf[i], norm_i ))
        norm.append(norm_i)
    print(norm)
    print(np.mean(norm))
    idx_discard = []
    for i, norm_i in enumerate(norm):
        if norm_i > 5* np.mean(norm):
            idx_discard.append(i)
    print(idx_discard)

    coor_src_match_new = [coor for i, coor in enumerate(coor_src_match) if i not in idx_discard]
    coor_tar_match_new = [coor for i, coor in enumerate(coor_tar_match) if i not in idx_discard]
    # print("Original matches:")
    # print(coor_src_match)
    # print("")
    # print(coor_tar_match)

    # print("New matches:")
    # print(coor_src_match_new)
    # print("")
    # print(coor_tar_match_new)

    # # Re-run rigid transformation but without outliers
    # R, t, s, _ = scaled_PCR(source = coor_src_match_new, target = coor_tar_match_new, max_iterations=50, plot_cost=visualize, verbose=verbose)
    # transformation_parameters = [R, t, s]
    # print("transformation parameters update:", transformation_parameters)
    # coor_src_match_new_tf = rigid_transform(coor_src_match_new, R, t, s, 'point')
    # # print(coor_src_match_new_tf)

    # Transfrom source point cloud
    point_coor_src_tf = rigid_transform(point_coor_src, R, t, s, 'point') 

    # Geometric verification: check if matched points are close in the common reference frame
    match_idx, _, _ = descriptor_matching(dist, r, verbose) # Less strict ratio test to allow for more matches
    matches = geometric_verification(match_idx, point_coor_tar, point_coor_src_tf, verbose) # correct matches based on coordinates
    print("matches after geometric verification:",  np.sum(~np.isnan(matches)))

    # TO DO: Discard points with too large norm from point cloud
    # for i, point in enumerate(point_coor_src_tf):
    #     print("point {}; coor = {}, norm = {}".format(i, point, np.linalg.norm(point)))

    if visualize:
        visualize_pointcloud(point_coor_src, point_coor_src_tf, point_coor_tar, 0, "Scaled Point Cloud Registration (entire pointcloud)")
        visualize_pointcloud(coor_src_match, coor_src_match_tf, coor_tar_match, 1, "Scaled Point Cloud Registration (confirmed matches)")
        # visualize_pointcloud(coor_src_match_new, coor_src_match_new_tf, coor_tar_match_new, 1, "Scaled Point Cloud Registration (confirmed matches, outlier rejection)")

    
    return transformation_parameters, matches

def nbr_matches_r(points, dist, r):
    # Extract point information for both sequences
    point_coor_tar, descr_tar, _, obsv_tar = information_points(points[0])
    point_coor_src, descr_src, _, obsv_src = information_points(points[1])

    match_idx, _, _ = descriptor_matching(dist, 0.3, 0) # Strict ratio test to ensure correct matches for transformations

    _, _, coor_tar_match, coor_src_match = point_matching(match_idx, obsv_tar, obsv_src, point_coor_tar, point_coor_src)



    # Transform point clouds to common reference frame
    R, t, s, _ = scaled_PCR(source = coor_src_match, target = coor_tar_match, max_iterations=50, plot_cost=0, verbose=0)

    # Transfrom source point cloud
    point_coor_src_tf = rigid_transform(point_coor_src, R, t, s, 'point') 

    # Geometric verification: check if matched points are close in the common reference frame
    match_idx, _, _ = descriptor_matching(dist, r, 0) # Less strict ratio test to allow for more matches
    matches_before = copy.deepcopy(match_idx)
    matches = geometric_verification(match_idx, point_coor_tar, point_coor_src_tf, 0) # correct matches based on coordinates

    return matches_before, matches

def visualize_nbr_matches_r(r, nbr_matches, sequences):
    fig, ax = plt.subplots(figsize=(16,9))
    fig_rel, ax_rel = plt.subplots(figsize=(16,9))

    colors = ['red', 'blue']
    for i, matches in enumerate(nbr_matches):
        # Absolute values
        ax.plot(r, matches[0], 'x--', linewidth=3, markersize = 10, color=colors[i])
        ax.plot(r, matches[1], 'o--', linewidth=3, markersize = 10, color=colors[i])

        # Relative values
        ax_rel.plot(r, matches[0]/matches[0][0]*100, 'x--', linewidth=3, markersize = 10, color=colors[i])
        ax_rel.plot(r, matches[1]/matches[0][0]*100, 'o--', linewidth=3, markersize = 10, color=colors[i])

    # Set legend elements
    legend_elements =  [Line2D([0], [0], marker='x', color='k', label = "Before geometric verification",
                          markerfacecolor='k', markersize=10, linewidth=0, markeredgecolor='k'),
                        Line2D([0], [0], marker='o', color='k', label = "After geometric verification",
                          markerfacecolor='k', markersize=10, linewidth=0)]
    
    for i in range(len(matches)):
        sequence_label =  '{} & {}'.format(sequences[i].split('_')[0],sequences[i].split('_')[1] )
        legend_elements.append(Line2D([0], [0], linestyle='--', color=colors[i], label = sequence_label))
    ax.legend(handles = legend_elements, labelspacing = 1, fontsize = 15)
    ax_rel.legend(handles = legend_elements, labelspacing = 1, fontsize = 15)


    # Set axis properties
    # ax.legend(fontsize = 16)
    ax.set_xlabel('r [-]', fontsize = 20)
    ax.set_ylabel('Number of matches [-]',fontsize = 20)
    ax.set_xticks(np.linspace(0,1,11))
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_title('Number of matches as a function of the acceptance threshold (absolute)',fontweight ='bold', fontsize= 24)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    ax.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')
    ax.invert_xaxis()


    ax_rel.set_xlabel('r [-]', fontsize = 20)
    ax_rel.set_ylabel('Number of matches [%]',fontsize = 20)
    ax_rel.set_xticks(np.linspace(0,1,11))
    ax_rel.tick_params(axis='both', which='major', labelsize=14)


    ax_rel.set_title('Number of matches as a function of the acceptance threshold (relative)',fontweight ='bold', fontsize= 24)
    ax_rel.minorticks_on()
    ax_rel.grid(which='major', linestyle='-', linewidth='0.25', color='dimgray')
    ax_rel.grid(which='minor', linestyle=':', linewidth='0.25', color='dimgray')
    ax_rel.invert_xaxis()

    plt.show()

    
## Helper functions
def int2bin(integer): # Convert integer to 8-bit binary vector
    """
    This function converts an integer to a vinary vector (8 bits).
    
    Arguments:
    - integer -- Integer to convert (max 255)

    Returns:
    - distances -- n x m matrix containing the respective distances, according to the chosen norm 
    """     
    if integer <= 255: 
        bin_str = bin(integer)[2:].zfill(8)
    else:
        raise ValueError("Integer is too large, cannot be represented by an 8-bit vector. Please only input integer values smaller than 256.")
    return [int(x) for x in bin_str]

def find_imgpath(directory, timestamp): # Find correct image to display matches
    images = os.listdir(directory)    
    partial_filename = str(float2int(timestamp)[:-1])
    for img in images: # iterate over the files to find the matching file
        if partial_filename in img:
            img_path = directory + img
            break
    return img_path

## Combine data for MABA
def point_transformation(tf_param, points, kf_id_offset): # Transform points into common reference system (correct format)
    # Extract relevant information from points
    point_coor, descriptors, frame_ids, observations = information_points(points)
    
    # Apply transformation to camera poses
    R, t, s = tf_param
    point_coor_tf = rigid_transform(point_coor, R, t, s, "point")

    # Recombine transformed point positions in points struct
    points_tf = []
    for coor, descr, kf_id, obsv in zip(point_coor_tf, descriptors, frame_ids, observations):
        point = [coor, descr]
        # Recombine frame_id + observation coordinates
        for i in range(len(kf_id)):
            obs = np.array([kf_id[i]+kf_id_offset, obsv[i][0], obsv[i][1], obsv[i][2], obsv[i][3]]) # Adapt kf id to avoid conflict when creating common data struct
            point.append(obs)

        points_tf.append(point)

    return points_tf

def keyframe_transformation(tf_param, keyframes): # Transform keyframes into common reference system (correct format)
    # Extract relevant information from keyframes
    kf_identifier, timestamp, cam_pos, cam_rot_vec, cam_rot_mat = information_keyframes(keyframes)
    
    # Apply transformation to camera poses
    R, t, s = tf_param

    P = []
    for i in range(len(keyframes)): # Convert camera poses to Camera matrix P
        R_cam = cam_rot_mat[i]
        t_cam = np.reshape(cam_pos[i], (3,1))
        P.append(np.vstack((np.hstack((R_cam, t_cam)), [0, 0, 0, 1])))

    cam_rot_mat_tf, cam_pos_tf = rigid_transform(P, R, t, s, "keyframe")
    cam_quat_tf = np.reshape([rotmat2quat(cam_rot, "xyzw") for cam_rot in cam_rot_mat_tf], (-1,4))


    # Recombine transformed poses in keyframes struct
    keyframes_tf = []
    for kf_id, time, cam_pos, cam_quat in zip(kf_identifier, timestamp, cam_pos_tf, cam_quat_tf):
        pos_x, pos_y, pos_z = cam_pos[0], cam_pos[1], cam_pos[2]
        quat_x, quat_y, quat_z, quat_w = cam_quat[0], cam_quat[1], cam_quat[2], cam_quat[3]
        kf = [kf_id, time, pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        keyframes_tf.append(kf)
    
    return keyframes_tf

def combine_kf_seq(keyframes): # Create one common struct for all keyframes of the different sequences
    """
    This function take an array containing the keyframes of multiple sequences and combines it into one list.
    Moreover, the keyframe id of the sequences are modified (kf_id + 1000*nbr_sequence) such that one can easily 
    differentiate between the different sequences.
    
    Arguments:
    - keyframes -- Array containing the keyframes for all sequences, extracted from the ORB-SLAM output files

    Returns:
    - seq -- Array that combines the keyframes of the different sequences, with adapted keyframe identifiers
    Note that the keyframes from the first sequence passed along in :keyframes: is used as baseline.
    """       

    # Initialize resulting list by copying first sequence
    seq = copy.deepcopy(keyframes[0])

    # Initialize loop
    for i in range(1, len(keyframes)):
        seq_i = keyframes[i]

        # Adapt kf_id to avoid conflict
        seq_kf_adapt = [np.insert(seq[1:], 0, seq[0] + 1000*i) for seq in seq_i]
        seq.extend(seq_kf_adapt)

    return seq

def combine_points_seq(points, match_idxs): #  Create one common struct for all points of the different sequences
   
    # Initialize resulting list by copying first sequence
    seq = copy.deepcopy(points[0])

    # Loop through match_idxs to find matches
    nbr_match = 0
    for i, match_idx in enumerate(match_idxs):
        if not np.isnan(match_idx): # match => add observations for other sequence
            nbr_match += 1
            match_point_obs = points[1][match_idx][2:]
            for match_obs in match_point_obs:
                seq[i].append(match_obs)
    print("Number of matches:", nbr_match)

    # Add points from sequence 2 that do not match with points sequence 1
    match_idxs_sort = np.sort(match_idxs) # Sort indices to facilitate search
    # print("Length of match_idxs_sort:", len(match_idxs_sort))
    match_idx_sort_unique = np.unique(match_idxs_sort)
    print("Length of match_idxs_sort_unique:", len(match_idx_sort_unique)-1)

    for i in range(len(points[1])):
        if i not in match_idx_sort_unique: # Point is only seen by sequence 2
            seq.append(points[1][i]) # Add points to seq
        else:
            match_idx_sort_unique = match_idx_sort_unique[1:] # remove idx from array to make search faster
    
    return seq


def combine_points_seq(points, match_idxs): #  Create one common struct for all points of the different sequences
   
    # Initialize resulting list by copying first sequence
    seq = copy.deepcopy(points[0])

    # Loop through match_idxs to find matches
    nbr_match = 0
    for i, match_idx in enumerate(match_idxs):
        if not np.isnan(match_idx): # match => add observations for other sequence
            nbr_match += 1
            match_point_obs = points[1][match_idx][2:]
            for match_obs in match_point_obs:
                seq[i].append(match_obs)
    print("Number of matches:", nbr_match)

    # Add points from sequence 2 that do not match with points sequence 1
    match_idxs_sort = np.sort(match_idxs) # Sort indices to facilitate search
    # print("Length of match_idxs_sort:", len(match_idxs_sort))
    match_idx_sort_unique = np.unique(match_idxs_sort)
    print("Length of match_idxs_sort_unique:", len(match_idx_sort_unique)-1)

    for i in range(len(points[1])):
        if i not in match_idx_sort_unique: # Point is only seen by sequence 2
            seq.append(points[1][i]) # Add points to seq
        else:
            match_idx_sort_unique = match_idx_sort_unique[1:] # remove idx from array to make search faster
    # Add observations of sequence 1 to the remaining points of sequence 2
    return seq












def create_sequence(keyframes, points, visualize, sequences): # Create resulting datastructs for keyframes and points
    """
    This function will create a common struct for both the keyframes and points containing the data of the provided sequences.
    To this end, all sequences are first transformed into a common reference frame. The rigid transformation parameters are estimated
    based on a strict keypoint matching.
    """
    tf_param, matches = keypoint_matching(points, 1, 0)

    # np.save('matches.npy', matches)

    # print("Tf params = ", tf_param)
    print("Check complete")
    keyframes_tf = keyframe_transformation(tf_param, keyframes[1])
    points_tf = point_transformation(tf_param, points[1], 1000)

    keyframes_combi = combine_kf_seq([keyframes[0], keyframes_tf])
    points_combi = combine_points_seq([points[0], points_tf], matches)

    if visualize:
        _, _, traj_org, _, _ = information_keyframes(keyframes_combi)
        plot_trajectories_combined(traj_org, len(keyframes[0]), sequences)

    return keyframes_combi, points_combi


def visual_check_matches(keyframes, points, matches, sequence_dir, save_path):
    # Extract information from keyframes and points
    kf_id1, timestamps_1, _, _, _ = information_keyframes(keyframes[0])
    kf_id2, timestamps_2, _, _, _ = information_keyframes(keyframes[1])
    _, _, frame_ids_1, observations_1 = information_points(points[0])
    _, _, frame_ids_2, observations_2 = information_points(points[1])

    dist_obs_1 = [[obs[2:] for obs in observation] for observation in observations_1]
    dist_obs_2 = [[obs[2:] for obs in observation] for observation in observations_2]

        # Struct with points that should be imaged
    points_to_visualize_seq1, points_to_visualize_seq2 = [], []
    for i, match in enumerate(matches):
        if not np.isnan(match): # the considered point has a match so it should be displayed
            points_to_visualize_seq1.append(i+500) # indices of points to visualize
            points_to_visualize_seq2.append(int(match))
    

        # Collect keyframe id + observation coordinates
    frame_ids_vis_1 = [frame_ids_1[idx] for idx in points_to_visualize_seq1] # frame ids of points to visualize
    frame_ids_vis_2 = [frame_ids_2[idx] for idx in points_to_visualize_seq2]

    obs_vis_1 =  [dist_obs_1[idx] for idx in points_to_visualize_seq1]  # pixels in frames of points to visualize
    obs_vis_2 =  [dist_obs_2[idx] for idx in points_to_visualize_seq2]


    # Visualize observations of points of sequence 1 that have a match with index
    for i in range(len(keyframes[0])):
        obs = []
        img_path = find_imgpath(sequence_dir[0], timestamps_1[i])

        
        # Find all points seen by cam0 at keyframe i
        kf_id = kf_id1[i] # number of keyframe
        # print("keyframe id:", kf_id)
        idx = [j for j, frame in enumerate(frame_ids_vis_1) if np.isin(kf_id, frame)] # indices of points from points_to_visualize that are observed by kf i
        # print("idx:", idx)
        for ind in idx:
            # print("ind:", ind)
            # print(frame_ids_vis_1[ind]) # frame ids of frames that observe the considered point
            # Find observed values from ORB-SLAM output files
            index_kf = np.where(frame_ids_vis_1[ind] == kf_id)[0][0]
            # print("index keyframe:", index_kf)
            observation = obs_vis_1[ind][index_kf]
            # print(observation)
            obs.append((observation[0], observation[1], points_to_visualize_seq1[ind])) # all observation values that must be visualized for keyframe i
            # obs.append((observation[0], observation[1], k)) # all observation values that must be visualized for keyframe i

        visualize_matching_observations(img_path, obs, kf_id, 1, save_path[0])
    
    # Visualize observations of matching points with same index   
    for i in range(len(keyframes[1])):
        obs = []
        img_path = find_imgpath(sequence_dir[1], timestamps_2[i])

        # Find all points seen by cam0 at keyframe i
        kf_id = kf_id2[i] # number of keyframe

        idx = [j for j, frame in enumerate(frame_ids_vis_2) if np.isin(kf_id, frame)] # indices of points from points_to_visualize that are observed by kf i
        for ind in idx:
            index_kf = np.where(frame_ids_vis_2[ind] == kf_id)[0][0]
            observation = obs_vis_2[ind][index_kf]
            obs.append((observation[0], observation[1], points_to_visualize_seq1[ind])) # all observation values that must be visualized for keyframe i


        visualize_matching_observations(img_path, obs, kf_id, 1, save_path[1])





## Currently not used
def match_points(path, descriptors, desciptors_db, seq_names, thresh):
    # Find  to which point in seq0 the points from sequence i match.
    idx_matches, sim_matches = [], []
    for i in range(len(desciptors_db)):
        idx_match, sim_match = descriptor_matching([descriptors[i]], [desciptors_db], thresh)
        idx_matches.append(idx_match[0][0]), sim_matches.append(sim_match[0][0])

    # save indices and similarities for later use
    np.save(path + 'idx_matches_{}_{}'.format(seq_names[0],seq_names[1]), idx_matches)
    np.save(path + 'sim_matches_{}_{}'.format(seq_names[0],seq_names[1]), sim_matches)

    return idx_matches, sim_matches

def match_points_similarity_check(idx_matches, sim_matches, threshold):
    for idx, sim in enumerate(sim_matches):
        if sim < threshold:
            idx_matches[idx] = np.nan
            sim_matches[idx] = np.nan
    
    return idx_matches, sim_matches

def nn_matching(source, target): # Finds the nearest neighbors between two point clouds
    """
    Finds the nearest neighbors between two point clouds.

    Arguments:
    - source -- the source point cloud
    - target -- the target point cloud

    Returns:
    - distances -- the distances between the nearest neighbors
    - indices -- the indices of the nearest neighbors in the target point cloud
    """
    distances = np.zeros(source.shape[0])
    indices = np.zeros(source.shape[0], dtype=int)

    for i, point in enumerate(source):
        differences = target - point
        distances[i] = np.min(np.linalg.norm(differences, axis=1))
        indices[i] = np.argmin(np.linalg.norm(differences, axis=1))

    return distances, indices

def dist2match(source, target): # Find distance between corresponding points
    differences = target - source
    return np.linalg.norm(differences, axis=1)

def main(keyframes, points, kf_id, visualize, dir, verbose):
    """
    First sequence: target
    Second sequence: source
    """
    # Extract keyframe and point information for both sequences

    kf_identifier_tar, timestamp_tar, cam_pos_tar, cam_rot_tar = information_keyframes(keyframes[0])
    kf_identifier_src, timestamp_src, cam_pos_src, cam_rot_src = information_keyframes(keyframes[1])

    point_coor_tar, descr_tar, frame_ids_tar, observations_tar = information_points(points[0])
    point_coor_src, descr_src, frame_ids_src, observations_src = information_points(points[1])

    # Estimate transformation parameters based on 2 input images using match_PointsInKf
    R, t, s = match_PointsInKf(keyframes, points, kf_id,  visualize, dir, verbose, parallel=1)
    
    # Use parameters to transform all keyframe camera poses and point coordinates of sequence 2 to match sequence 1
    point_coor_src_tf, cam_rot_src_tf, cam_pos_src_tf = transform_sequences(point_coor_src, cam_pos_src, cam_rot_src, R, t, s)
    visualize_pointcloud(point_coor_src, point_coor_src_tf, point_coor_tar, 0)

    # Perform aided descriptor matching based on tranformed point coordinates
    start = time.time() 
    dist = parallel_hamming_distance(descr_tar, descr_src)
    end = time.time()
    print("Elapsed time (parallel):", end - start, "seconds")  

    # descriptor_matching_aided(dist, point_coor_tar, point_coor_src)
    match_idx, min_dist, ratio_min = descriptor_matching(dist, 0.5)
    print('i','pt', '        minimal distances        ', 'ratio')
    print('-------------------------------------------------------')

    for i in range(len(match_idx)):
        if not np.isnan(match_idx[i]):
            print(i+1, match_idx[i], min_dist[i], 'r=',ratio_min[i], 'L2=',np.linalg.norm(point_coor_tar[i] - point_coor_src_tf[match_idx[i]]),
                  'L2_r=',np.linalg.norm(point_coor_tar[i] - point_coor_src_tf[match_idx[i]])/np.linalg.norm(point_coor_tar[i]),
                  'point tar:', point_coor_tar[i], 'point src:', point_coor_src_tf[match_idx[i]])
    print('-------------------------------------------------------')
    print("")


################################       Test functions       ################################

# ORB_version = 'ORB2'
# sequence = 'MH01'
# keyframes_MH01, points_MH01 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Regular/output_{}_MH01.txt'.format(ORB_version, ORB_version))
# keyframes_MH02, points_MH02 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Regular/output_{}_MH02.txt'.format(ORB_version, ORB_version))
# keyframes_MH03, points_MH03 = parsedata('/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/Regular/output_{}_MH03.txt'.format(ORB_version, ORB_version))


# kf_id = [610,63]
# # kf_id = [474, 471]
# directory = ['/home/pb/Documents/EUROC dataset/ASL/MH_01_easy/mav0/cam0/data/', '/home/pb/Documents/EUROC dataset/ASL/MH_02_easy/mav0/cam0/data/']



# keyframes_combi, points_combi = create_sequence([keyframes_MH01, keyframes_MH02], [points_MH01, points_MH02], 1)

# point_coors_no_noise, _, frame_ids, observations = information_points(points_MH01)
# print(observations)


