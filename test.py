import denclue
import dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

datas = dataset.create_3_points_dataset()

clusters = denclue.train_denclue_clustering(datas, h=0.004)

x = []
y = []

for cluster in clusters:
    x.append(cluster[0][0])
    y.append(cluster[0][1])

plt.plot(x, y, "o")
plt.plot(datas[:,0], datas[:,1], "o")
plt.show()