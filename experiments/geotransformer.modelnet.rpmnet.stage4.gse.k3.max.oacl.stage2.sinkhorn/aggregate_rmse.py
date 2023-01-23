import re
import matplotlib.pyplot as plt
import numpy as np

data_type = 'full_deformed'
training_data = 'pretrained'

model_numbers = ['002', '042', '085', '126', '167', '207']
filename = 'output_geo_' + data_type + '_td_' + training_data + '.txt'
file = open(filename, 'r')
lines = file.readlines()
current_model = None
final_data = {model_number : 0.0 for model_number in model_numbers}

for line in lines:
    if 'model' in line and len(line) < 100:        
        words = line.split(' ')
        current_model = words[1]
    
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
plt.savefig('output_geo_' + data_type + '_td_' + training_data + '.png')

