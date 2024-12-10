import json
import numpy as np
import glob
import os

files = "output_panda_fixed_arm"
motion_full =[]
name_full = []
for i, file in enumerate(os.listdir(files)):
    name_full.append(file.split('.')[0])
    motion = np.loadtxt(os.path.join(files,file), delimiter=',')
    motion_full.append(motion)

for j in range(len(motion_full)):
    json_data={
        'frame_duration':1/50,
        'frames':motion_full[j].tolist()
    }
    with open('output_panda_fixed_arm_json/'+name_full[j]+'.json', 'w') as f:
        json.dump(json_data, f, indent=4)

# file = "swing_2.txt"
# name = file.split('.')[0]
# motion = np.loadtxt(os.path.join(files,file), delimiter=',')
# json_data={
#     'frame_duration':1/50,
#     'frames':motion.tolist()
# }
# with open('output_json/'+'swing'+'.json', 'w') as f:
#     json.dump(json_data, f, indent=4)
