from evo.tools import file_interface
lib_location = "/home/pb/Documents/Thesis/Scripts/lib"
import sys
sys.path.insert(0, lib_location)

from functions import parsedata, save_keyframes2csv

# Save data as csv file
ORB_version = "ORB2"
kf_path = '/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Output/No GBA/'.format(ORB_version)
save_path = '/home/pb/Documents/Thesis/Data/ORB SLAM output/{}/Keyframes/No GBA/'.format(ORB_version)

keyframes_MH01, points_MH01 = parsedata(kf_path + 'MH01_no_GBA.txt'.format(ORB_version))
keyframes_MH02, points_MH02 = parsedata(kf_path + 'MH02_no_GBA.txt'.format(ORB_version))
keyframes_MH03, points_MH03 = parsedata(kf_path + 'MH03_no_GBA.txt'.format(ORB_version))
keyframes_MH04, points_MH04 = parsedata(kf_path + 'MH04_no_GBA.txt'.format(ORB_version))
keyframes_MH05, points_MH05 = parsedata(kf_path + 'MH05_no_GBA.txt'.format(ORB_version))

# keyframes_MH01, points_MH01 = parsedata(kf_path + 'output_{}_MH01.txt'.format(ORB_version))
# keyframes_MH02, points_MH02 = parsedata(kf_path + 'output_{}_MH02.txt'.format(ORB_version))
# keyframes_MH03, points_MH03 = parsedata(kf_path + 'output_{}_MH03.txt'.format(ORB_version))
# keyframes_MH04, points_MH04 = parsedata(kf_path + 'output_{}_MH04.txt'.format(ORB_version))
# keyframes_MH05, points_MH05 = parsedata(kf_path + 'output_{}_MH05.txt'.format(ORB_version))

save_keyframes2csv(save_path + 'CSV/{}_keyframes_MH01.csv'.format(ORB_version), keyframes_MH01)
save_keyframes2csv(save_path + 'CSV/{}_keyframes_MH02.csv'.format(ORB_version), keyframes_MH02)
save_keyframes2csv(save_path + 'CSV/{}_keyframes_MH03.csv'.format(ORB_version), keyframes_MH03)
save_keyframes2csv(save_path + 'CSV/{}_keyframes_MH04.csv'.format(ORB_version), keyframes_MH04)
save_keyframes2csv(save_path + 'CSV/{}_keyframes_MH05.csv'.format(ORB_version), keyframes_MH05)

# convert csv file of estimated poses to TUM trajectory
kf_data_MH01 = file_interface.read_euroc_csv_trajectory(save_path + 'CSV/{}_keyframes_MH01.csv'.format(ORB_version))
kf_data_MH02 = file_interface.read_euroc_csv_trajectory(save_path + 'CSV/{}_keyframes_MH02.csv'.format(ORB_version))
kf_data_MH03 = file_interface.read_euroc_csv_trajectory(save_path + 'CSV/{}_keyframes_MH03.csv'.format(ORB_version))
kf_data_MH04 = file_interface.read_euroc_csv_trajectory(save_path + 'CSV/{}_keyframes_MH04.csv'.format(ORB_version))
kf_data_MH05 = file_interface.read_euroc_csv_trajectory(save_path + 'CSV/{}_keyframes_MH05.csv'.format(ORB_version))

file_interface.write_tum_trajectory_file(save_path + 'TUM/{}_kf_MH01_TUM.txt'.format(ORB_version), kf_data_MH01)
file_interface.write_tum_trajectory_file(save_path + 'TUM/{}_kf_MH02_TUM.txt'.format(ORB_version), kf_data_MH02)
file_interface.write_tum_trajectory_file(save_path + 'TUM/{}_kf_MH03_TUM.txt'.format(ORB_version), kf_data_MH03)
file_interface.write_tum_trajectory_file(save_path + 'TUM/{}_kf_MH04_TUM.txt'.format(ORB_version), kf_data_MH04)
file_interface.write_tum_trajectory_file(save_path + 'TUM/{}_kf_MH05_TUM.txt'.format(ORB_version), kf_data_MH05)

