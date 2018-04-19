
import os
import time
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import csv
#subprocess.call('python KNN.py',shell=True)
#subprocess.call('python Decision_Tree.py',shell=True)
# subprocess.call('python KNN_Varying_K.py',shell=True)


with open('I:/Masters/SPRING 18/SPATIAL AND TEMPORAL/spatial/homework 2/before feature sel/KNN-Algorithm/KNN_Average_K_Accuracy.csv') as f1:
    r=csv.reader(f1,delimiter=',')
    for row in r:
        a=row
        next(r)
    for x in range(len(a)):
        a=[int(x) for x in a]
#
# with open('KNN_Varying_K_Accuracy.csv') as f1:
#     r=csv.reader(f1,delimiter=',')
#     for row in r:
#         b=row
#         next(r)
#     for x in range(len(b)):
#         b=[int(x) for x in b]

with open('I:/Masters/SPRING 18/SPATIAL AND TEMPORAL/spatial/homework 2/before feature sel/Decision-Tree_train/DecisionTree_Accuracy.csv') as f2:
    r=csv.reader(f2,delimiter=',')
    for row in r:
        c=row
        next(r)
    for x in range(len(c)):
        c=[int(x) for x in c]
# print(b)

# data to plot
n_groups = 5
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8

rects1 = plt.bar(index, a, bar_width,
                 alpha=opacity,
                 color='b',
                 label='KNN')
# rects2 = plt.bar(index + bar_width, b, bar_width,
#                  alpha=opacity,
#                  color='r',
#                  label='KNN_Varying_K')
rects3 = plt.bar(index + bar_width, c, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Decision Tree')


plt.xlabel('Folds')
plt.ylabel('Accuracy')
plt.title('Accuracy by Algorithm')
plt.xticks(index + bar_width, ('1', '2', '3', '4','5'))
plt.legend()

plt.tight_layout()
plt.show()

