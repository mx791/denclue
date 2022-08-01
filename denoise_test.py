import denclue
import dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

datas = dataset.create_3_points_dataset()
datas = dataset.add_noise(datas, 20)
model = denclue.train_denclue_clustering(datas, h=0.005, noise_treshold=0.1)

x = []
y = []

for cluster in model.getClusters():
    x.append(cluster[0])
    y.append(cluster[1])

filtered_datas = []
for line in datas:
    if not model.isNoise(line):
        filtered_datas.append(line)

filtered_datas = np.array(filtered_datas)

plt.plot(datas[:,0], datas[:,1], "o")
plt.plot(filtered_datas[:,0], filtered_datas[:,1], "o")
plt.plot(x, y, "o")
plt.show()