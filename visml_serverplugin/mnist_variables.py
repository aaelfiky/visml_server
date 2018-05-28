import csv
import numpy as np
import os


os.chdir(r'/home/ahmed/studentProject/workspace/MNIST_BK/visml_serverplugin/visml_serverplugin')
cwd = os.path.dirname(os.path.realpath(__file__))
print(cwd)

with open('mnist_test.csv', 'rb') as f:
    reader = csv.reader(f)
    train_data_list = list(reader)
    train_data = np.array(train_data_list)

print(train_data[0])
