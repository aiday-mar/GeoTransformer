import re
import matplotlib.pyplot as plt
import numpy as np

# data_type = 'partial_non_deformed'
data_type = 'full_non_deformed'

training_data = 'pretrained'
# training_data = 'full_non_deformed'
# training_data = 'partial_non_deformed'

current_deformation = True
# current_deformation = False

if current_deformation is True:
    init_voxel_size = 0.01
else:
    init_voxel_size = 0.05

model_numbers = ['002', '042', '085', '126', '167', '207']
if current_deformation is False:
    filename = 'output_geo_' + data_type + '_td_' + training_data + '_ivs_' + str(init_voxel_size) + '.txt'
else:
    filename = 'output_geo_' + data_type + '_td_' + training_data + '_ivs_' + str(init_voxel_size) + '_current_deformation.txt'

file = open(filename, 'r')
lines = file.readlines()
current_model = None
final_data = {model_number : 0.0 for model_number in model_numbers}

for line in lines:
    if 'model' in line and len(line) < 100:        
        words = line.split(' ')
        current_model = words[1]
        
        if data_type == 'full_non_deformed' or data_type == 'partial_non_deformed':
            current_model = current_model[:len(current_model)-1]
    
    if 'RMSE' in line and current_model:
        list_res = re.findall("\d+\.\d+", line)
        rmse = float(list_res[0])
        final_data[current_model] = rmse

rmse = []
for model_number in model_numbers:
    rmse.append(final_data[model_number])

bar = np.array([0, 1, 2, 3, 4, 5])
plt.bar(bar, rmse)
plt.xticks(bar, model_numbers)
plt.xlabel("Model number")
plt.ylabel("RMSE")
data_type_mod = data_type.replace('_', ' ').title()
plt.title(data_type_mod)

if current_deformation is False:
    image_filename = 'output_geo_' + data_type + '_td_' + training_data + '_ivs_' + str(init_voxel_size) + '.png'
else:
    image_filename = 'output_geo_' + data_type + '_td_' + training_data + '_ivs_' + str(init_voxel_size) + '_current_deformation.png'

plt.savefig(image_filename)

