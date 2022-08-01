import math
import random
import numpy as np
import matplotlib.pyplot as plt

# créer un jeu de données avec 3 clusters
def create_3_points_dataset():
    datas = []
    for i in range(50):
        datas.append([random.random()*0.3, 0.5+random.random()*0.4])
    for i in range(50):
        datas.append([0.6+random.random()*0.2, 0.5+random.random()*0.3])
    for i in range(50):
        datas.append([0.5+random.random()*0.3, random.random()*0.4])
    return np.array(datas)

# créer un jeu de données avec 2 clusters -> cercles concentriques
def concentric_circles():
    datas = []
    for i in range(50):
        datas.append([0.3 - random.random()*0.3, 0.3 - random.random()*0.3])
    while len(datas) < 300:
        vect = [random.random()*2 - 1, random.random()*2- 1]
        sum = vect[0]**2 +  vect[1]**2
        if sum > 0.6 and sum < 0.9:
            datas.append(vect)
    return np.array(datas)

#datas = create_3_points_dataset()
datas = concentric_circles()

plt.plot(datas[:,1], datas[:,0], "o")
plt.show()