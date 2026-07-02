import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.cluster import MiniBatchKMeans, KMeans
import gpxpy.geo
from datetime import datetime, timedelta
from joblib import dump, load


def min_distance(regionCenters, totalClusters):
    less_dist = []
    more_dist = []
    min_distance = np.inf  # any big number can be given here
    for i in range(totalClusters):
        good_points = 0
        bad_points = 0
        for j in range(totalClusters):
            if j != i:
                distance = gpxpy.geo.haversine_distance(latitude_1=regionCenters[i][0], longitude_1=regionCenters[i][1],
                                                        latitude_2=regionCenters[j][0], longitude_2=regionCenters[j][1])
                # distance from meters to miles
                distance = distance / (1.60934 * 1000)
                # it will return minimum of "min_distance, distance".
                min_distance = min(min_distance, distance)
                if distance < 2:
                    good_points += 1
                else:
                    bad_points += 1
        less_dist.append(good_points)
        more_dist.append(bad_points)
    print("On choosing a cluster size of {}".format(totalClusters))
    print("Avg. Number clusters within vicinity where inter cluster distance < 2 miles is {}".format(
        np.ceil(sum(less_dist) / len(less_dist))))
    print("Avg. Number clusters outside of vicinity where inter cluster distance > 2 miles is {}".format(
        np.ceil(sum(more_dist) / len(more_dist))))
    print("Minimum distance between any two clusters = {}".format(min_distance))
    print("-" * 10)


def makingRegions(noOfRegions, coord):
    regions = MiniBatchKMeans(n_clusters=noOfRegions,
                              batch_size=10000, random_state=0).fit(coord)
    regionCenters = regions.cluster_centers_
    totalClusters = len(regionCenters)
    return regionCenters, totalClusters


def optimal_cluster(df, coord):
    startTime = datetime.now()
    for i in range(10, 100, 10):
        regionCenters, totalClusters = makingRegions(i, coord)
        min_distance(regionCenters, totalClusters)
    print("Time taken = " + str(datetime.now() - startTime))
