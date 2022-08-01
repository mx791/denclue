import math
import random
import numpy as np
import matplotlib.pyplot as plt

class KernelDensityEstimator:

    def __init__(self, datas, sigma=1, n_samples=100):
        self.datas = datas
        self.sigma = sigma
        self.n_samples = min(len(datas), n_samples)

    # calcul la densité en un point donné
    def getDensity(self, point_coordinates):
        # TODO
        pass

    # calcul le gradient de la fonction de densité
    def getDensityGradient(self, point_coordinates):
        # TODO
        pass

# ascension de gradient pour trouver un maximum local de la fonction de densité
def gradient_ascent(density_estimator, starting_point):
    # TODO
    return [starting_point, starting_point]

# process d'entrainement
def train_denclue_clustering(datas, sigma=1, n_samples=100):

    clusters = [{Center: np.array([np.random(), np.random()]), LastCenter: np.array([np.random(), np.random()])}]

    return DenclueCustering(clusters, KernelDensityEstimator(datas))

class DenclueCustering:

    def __init__(self, clusters, density_estimator):
        self.clusters = clusters
    
    # la densité en un point est elle suffisante, ou s'agit-il de bruit ?
    def isNoise(point_coordinates):
        # TODO
        return False
    
    def getCluster(self, point_coordinates):
        # TODO
        return 0