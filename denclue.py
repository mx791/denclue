import math
import random
from select import select
import numpy as np
import matplotlib.pyplot as plt



class KernelDensityEstimator:

    def __init__(self, datas, h=0.01, n_samples=100):
        self.datas = datas
        self.h = h
        self.n_samples = min(len(datas), n_samples)

    # calcul la densité en un point donné
    def getDensity(self, point_coordinates):
        density = 0.0
        for point in self.datas:
            delta = np.sum((point - point_coordinates)**2)**2
            #density += 1 / (2* np.pi * self.sigma**2) * np.exp(-delta/(2*self.sigma**2))
            #density += (delta + 0.5) * (0.5 - delta)
            density += np.exp(-(delta/self.h)**2) / ((2.*np.pi)**(1/2))
        return density

    # calcul le gradient de la fonction de densité
    def getDensityGradient(self, point_coordinates):
        gradient = []
        h = 0.01
        for i in range(len(point_coordinates)):
            point_coordinates[i] += h
            value_before = self.getDensity(point_coordinates)
            point_coordinates[i] -= h*2
            value_after = self.getDensity(point_coordinates)
            point_coordinates[i] += h
            gradient.append(value_before - value_after)
        return gradient

def gradient_stepp(grad, lr):
    if grad > 0:
        return lr
    else:
        return -lr

# ascension de gradient pour trouver un maximum local de la fonction de densité
def gradient_ascent(density_estimator, starting_point):
    current_density = density_estimator.getDensity(starting_point)
    last_density = 0
    current_point = starting_point.copy()
    last_point = starting_point.copy()
    points = []

    lr = 0.01
    counter = 0
    while current_density >= last_density:
        gradient = density_estimator.getDensityGradient(current_point)
        last_point = current_point.copy()
        for i in range(len(gradient)):
            current_point[i] += gradient_stepp(gradient[i], lr)
        last_density = current_density
        points.append(current_point.copy())
        current_density = density_estimator.getDensity(current_point)
        counter += 1
        if counter > 75:
            break

    # print(last_density, density_estimator.getDensity(starting_point))

    return [last_point, starting_point]



# process d'entrainement
def train_denclue_clustering(datas, h=0.1, n_samples=100, noise_treshold=0.1):

    clusters = []
    estimator = KernelDensityEstimator(datas, h=h)

    # on trouve le point d'attirance de chaque ligne, et créer autant de clusters
    for point in datas:
        clusters.append(gradient_ascent(estimator, point))
        print(len(clusters), "/", len(datas))

    # on supprime les clusters redondants
    #TODO

    return np.array(clusters)

    return DenclueCustering(clusters, KernelDensityEstimator(datas))




class DenclueCustering:

    def __init__(self, clusters, density_estimator, treshold):
        self.clusters = clusters
        self.estimator = density_estimator
        self.treshold = treshold
    
    # la densité en un point est elle suffisante, ou s'agit-il de bruit ?
    def isNoise(self, point_coordinates):
        return self.estimator.getDensity(point_coordinates) > self.treshold
    
    def getCluster(self, point_coordinates):
        # TODO
        return 0