import denclue
import dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

datas = dataset.create_3_points_dataset()

model = denclue.train_denclue_clustering(datas, h=0.005)

x = []
y = []

for cluster in model.getClusters():
    x.append(cluster[0])
    y.append(cluster[1])


plt.plot(datas[:,0], datas[:,1], "o")
plt.plot(x, y, "o")
plt.show()