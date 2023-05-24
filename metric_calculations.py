import sys
lib_location = "/home/pb/Documents/Thesis/Scripts/lib"
sys.path.insert(0, lib_location)

from evo_metrics import calculate_metrics

################################ Run ################################
type_traj = 'before'
ORB_type = 'No GBA'
ORB_version = 'ORB2'
sequences = ['MH01','MH02','MH03','MH04','MH05']
sequences = ['MH01']
Th_val = 'Th_1'

APE_before_BA, RPE_before_BA = calculate_metrics('before', ORB_type, ORB_version, sequences, Th_val, "APE", save=0, figures=1)
# APE_after_BA, RPE_after_BA = calculate_metrics('after', ORB_type, ORB_version, sequences, Th_val, "both", save=0, figures=0)

print("APE translation:")
print("Before BA:", APE_before_BA[0])
# print("After BA:", APE_after_BA[0])

print("\nAPE rotation:")
print("Before BA:", APE_before_BA[1])
# print("After BA:", APE_after_BA[1])

# print("\nAPE full:")
# print("Before BA:", APE_before_BA[2])
# print("After BA:", APE_after_BA[2])

# print("\nRPE translation:")
# print("Before BA:", RPE_before_BA[0])
# print("After BA:", RPE_after_BA[0])

# print("\nRPE rotation:")
# print("Before BA:", RPE_before_BA[1])
# print("After BA:", RPE_after_BA[1])

# print("\nRPE full:")
# print("Before BA:", RPE_before_BA[2])
# print("After BA:", RPE_after_BA[2])





