from evo.tools import file_interface

ORB_version = 'ORB2'
ORB_type = 'No GBA'
BA_path = '/home/pb/Documents/Thesis/Data/SABA/{}/{}'.format(ORB_version, ORB_type)
Th = 'Th_1'

# convert csv file of estimated poses to TUM trajectory
BA_kf_data_MH01 = file_interface.read_euroc_csv_trajectory(BA_path + '/CSV/' + Th + '/BA_kf_MH01.csv')
BA_kf_data_MH02 = file_interface.read_euroc_csv_trajectory(BA_path + '/CSV/' + Th + '/BA_kf_MH02.csv')
BA_kf_data_MH03 = file_interface.read_euroc_csv_trajectory(BA_path + '/CSV/' + Th + '/BA_kf_MH03.csv')
BA_kf_data_MH04 = file_interface.read_euroc_csv_trajectory(BA_path + '/CSV/' + Th + '/BA_kf_MH04.csv')
BA_kf_data_MH05 = file_interface.read_euroc_csv_trajectory(BA_path + '/CSV/' + Th + '/BA_kf_MH05.csv')

file_interface.write_tum_trajectory_file(BA_path + '/TUM/' + Th + '/BA_kf_MH01_TUM.txt', BA_kf_data_MH01)
file_interface.write_tum_trajectory_file(BA_path + '/TUM/' + Th + '/BA_kf_MH02_TUM.txt', BA_kf_data_MH02)
file_interface.write_tum_trajectory_file(BA_path + '/TUM/' + Th + '/BA_kf_MH03_TUM.txt', BA_kf_data_MH03)
file_interface.write_tum_trajectory_file(BA_path + '/TUM/' + Th + '/BA_kf_MH04_TUM.txt', BA_kf_data_MH04)
file_interface.write_tum_trajectory_file(BA_path + '/TUM/' + Th + '/BA_kf_MH05_TUM.txt', BA_kf_data_MH05)

print(BA_kf_data_MH01)

