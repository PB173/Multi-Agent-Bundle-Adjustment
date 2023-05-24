PyCeres_location = "/home/pb/ceres-bin/lib"
import sys
sys.path.insert(0,PyCeres_location)
import PyCeres  # Import the Python Bindings

lib_location = "/home/pb/Documents/Thesis/Scripts/lib"
sys.path.insert(0, lib_location)

from rotation_transformations import quat2rotmat, rotmat2quat
from evo.tools import file_interface

import numpy as np
import pandas as pd
import re

# Parse input file to extract the relevant data
def parsedata(inputfile):
    """
    Parse data provided in input file to extract the keyframes and the points captured by ORB-SLAM.
    
    Input
    :param inputfile: .txt file generated with the modified ORB-SLAM (2 or 3) algorithm. This file should indicate '## keyframes' at line 1 and denote the start and end
                       of every point by "## point" and "## end point" respectively.
    
    Output
    :keyframes: array containing for every keyframe the key_id, timestamp, 3 translation parameters and a rotation quaternion.
    :points: array containing for every keyframe 3 position coordinates, 32 descriptors and, for every keyframe on which the point is observed, a key_id and
             2 observation parameters.
    """
    kf, points = [], []

    with open(inputfile) as f:
        line = f.readline()
        while line:
            if  "## keyframe" in line: #the following lines are keyframes
                line = f.readline()
                while "## points" not in line:
                    kf.append(line)
                    line = f.readline()
            
            if "## points" in line: #the following lines are points
                line = f.readline()
                while line:
                    if "# point" in line:
                        pt = []
                        line = f.readline()
                        while "# end point" not in line:
                            pt.append(line)
                            line = f.readline()
                        point = [np.array(re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", pt[i])).astype(float) for i in range(len(pt))]
                        points.append(point)
                    line = f.readline()

    keyframes = [np.array(re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", kf[i])).astype(float) for i in range(len(kf))]
    
    return keyframes, points

# Extract information from parsed input files
def information_points(points): # Extract information for all points
    """
    Extract information on all points of a sequence.
    """
    point_coor = np.reshape([np.array(point[0]) for point in points], (-1, 3))
    descr = []
    for point in points:
        descr.append([int(p) for p in point[1]])
    descr = np.reshape(descr,(-1,32))
    frame_ids = [np.array([(point[obs+2][0]).astype(int) for obs in range(len(point)-2)]) for point in points]
    observations = [np.array([(point[obs+2][1:]) for obs in range(len(point)-2)]) for point in points]

    return point_coor, descr, frame_ids, observations

def information_keyframes(keyframes): # Extract information for all keyframes
    """
    Extract information on all keyframes of a sequence.
    """
    kf_identifier = [keyframe[0].astype(int) for keyframe in keyframes]
    timestamp = [keyframe[1] for keyframe in keyframes]
    cam_pos = np.reshape([np.array(keyframe[2:5]) for keyframe in keyframes], (-1, 3))
    cam_rot_vec = np.reshape([np.array(quat2rotmat(keyframe[5:], "vector")) for keyframe in keyframes], (-1,9))
    cam_rot_mat = [np.array(quat2rotmat(keyframe[5:], "matrix")) for keyframe in keyframes]

    return kf_identifier, timestamp, cam_pos, cam_rot_vec, cam_rot_mat

def extract_point(points, kf_id): # Extract information for a specific point
    frame_ids = [np.array([(point[obs+2][0]).astype(int) for obs in range(len(point)-2)]) for point in points]
    points_kf = [p for p, f in zip(points, frame_ids) if kf_id in f]
    point_coor = [point[0] for point in points_kf] # coordinates of all points seen in the considered keyframe
    descr = [] # descriptors of all points seen in the considered keyframe
    for point in points_kf:
        descr.append([int(p) for p in point[1]])
    descr = np.reshape(descr,(-1,32))
    observations = np.reshape([obs[1:] for point in points_kf for obs in point[2:] if obs[0] == kf_id], (-1,4))

    # return frame_ids, points_kf, coor, descr, obsv
    return point_coor, descr, frame_ids, observations, points_kf


# Save data to txt file
def cam_pose2txt(keyframes, path):
    cam_pos = np.reshape([np.array(keyframe[2:5]) for keyframe in keyframes], (-1, 3))
    cam_rot = np.reshape([np.array(keyframe[5:]) for keyframe in keyframes], (-1,4))
    # cam_rot = cam_rot[:, [3, 0, 1, 2]]

    save_traj = np.concatenate((cam_pos, cam_rot), axis=1)

    # Save data to csv file
    np.savetxt(path, save_traj.astype(np.float64), delimiter=" ", fmt='%.5f' )

def save_kfBA2csv(path, keyframes, traj_BA, rot_BA, rot_order):
    # convert rotation matrix back to quaternion
    quat = [rotmat2quat(rot, rot_order) for rot in rot_BA]

    # concatenate data into one data construction
    dims = (-1, 1) # reshape timestamps into column vector
    time = np.reshape(np.array([keyframe[1]for keyframe in keyframes]), dims)*1e9 # to match with timestamps ground truth

    save_traj = np.concatenate((time, traj_BA, quat), axis=1)
    
    # Save data as csv file
    np.savetxt(path, save_traj, delimiter=",")

def save_noisy_kf2csv(path, keyframes, traj_noisy, rot_noisy, rot_order):
    # Extract relevant data
    time = np.reshape(np.array([keyframe[1]for keyframe in keyframes]), (-1, 1))*1e9 # to match with timestamps ground truth
    # quat = np.reshape([keyframe[5:] for keyframe in keyframes], (-1,4))

    quat = [rotmat2quat(rot, rot_order) for rot in rot_noisy]

    # if rot_order == "wxyz":
    #     quat_wxyz = []
    #     for i in range(len(quat)):
    #         quat_reorder = np.concatenate((quat[i][-1], quat[i][0:-1]), axis=None)
    #         quat_wxyz.append(quat_reorder)
        
    #     quat = quat_wxyz
    # concatenate data into one data construction
    save_traj = np.concatenate((time, traj_noisy, quat), axis=1)
    
    # Save data as csv file
    np.savetxt(path, save_traj, delimiter=",")

def save_pointsBA2csv(path, points_BA):
    np.savetxt(path, points_BA, delimiter=",")

def save_keyframes2csv(path, keyframes):
    time = np.reshape(np.array([keyframe[1]for keyframe in keyframes]), (-1, 1))*1e9 # to match with timestamps ground truth
    camera_translations = np.reshape([np.array(keyframe[2:5]) for keyframe in keyframes], (-1, 3))
    camera_rotations = np.reshape([np.array(keyframe[5:]) for keyframe in keyframes], (-1,4))
    save_traj = np.concatenate((time, camera_translations, camera_rotations), axis=1)
    np.savetxt(path, save_traj, delimiter=",")

def save_groundtruth2csv(path, groundtruth):
    np.savetxt(path, groundtruth, delimiter=",")

def save2csv(path, data, format='column'):
    """ Function used to automatically save a data array to a csv file. By defining format as either 'row' or 'column', 
    the data is saved in a row or column vector."""
    if format =='row':
        np.savetxt(path, np.array([data]), delimiter=",", fmt = "%.3f")
    else:
        np.savetxt(path, data, delimiter=",", fmt = "%.3f")

def saveARPE2csv(APE_RPE, stats, sequence, save_path = ''):
    if save_path == '':
        save_path = "/home/pb/Documents/Thesis/Data/Evo metrics/ORB-SLAM output/{}/".format(sequence) + APE_RPE + "_stats_" + sequence + ".csv"
    headers = ['rmse', 'mean', 'median', 'std', 'min', 'max', 'sse']
    result = np.array([0]*len(stats[0]))
    for i in range(len(stats)):
        data = []
        for h in headers:
            data.append(stats[i][h])
        result = np.vstack([result, data])

    save2csv(save_path, result[1:])

# Convert CSV to TUM
def csv2tum(csv_path, tum_path):
    csv_data = file_interface.read_euroc_csv_trajectory(csv_path)
    file_interface.write_tum_trajectory_file(tum_path, csv_data)

# Read ground truth as TUM file
def readfromTUM(path):
    return file_interface.read_tum_trajectory_file(path)

# Read BA output from csv file
def readBAoutput(path, type: str):
    data = pd.read_csv(path, header=None)
    if type == "keyframes":
        cam_transl = np.array(data.iloc[:,1:4].values)
        cam_rot = np.array(data.iloc[:,4:8].values)
    
        return cam_transl, cam_rot
    elif type == "points":
        return np.array(data.values)


# Convert floating point values to integers by simply removing the decimal point
def float2int(float):
    # concatenate the integer and the decimal digits
    return str(int(float)) + str(float)[str(float).index('.')+1:]


# Test cases

# path = '/home/pb/Documents/Thesis/Data/SABA/ORB2/CSV/BA_kf_MH01.csv'
# print(readBAoutput(path, "keyframes")[1][0])