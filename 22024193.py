# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:04:33 2023

@author: MUSTANG
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
from sklearn.cluster import KMeans
import cluster_tools as ct
import importlib

importlib.reload(ct)
import errors as err

df_gdp = pd.read_csv("API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5358515.csv", skiprows =(4))
df_emission = pd.read_csv('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5358347.csv', skiprows = 3)

def cluster(data):
    
    import sklearn.cluster as cluster
    import sklearn.cluster as cluster
    # reading data
    df_emission = pd.read_csv('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5358347.csv', skiprows = 3)
    print(df_emission)
    
    df_gdp = pd.read_csv("API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5358515.csv", skiprows=(4))
    print(df_gdp.describe())
    
    # cleaning the data and transposing
    df_emission = df_emission.drop(['1960','1961','1962','1963','1964','1965','1966','1967','1968','1969','1970','1971','1972','1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','Unnamed: 66'], axis=1)
    
    # transposing
    df = df_emission.transpose()
    
    #using logical slicing
    df_emission = df_emission[['Country Name','1990','2000','2010','2019']]
    df_emission = df_emission.dropna()
    print(df_emission)
    

    # transposing gdp data
    df2 = df_gdp.transpose()
    # working gdp data, cleaning and slicing
    df_gdp = df_gdp[['Country Name', '1990', '2000','2010', '2019']]
    df_gdp = df_gdp.dropna()
    print(df_gdp)
    
    #making a copy ,year 2010
    df_emission2019 = df_emission[["Country Name", "2019"]].copy()
    df_gdp2019 = df_gdp[["Country Name", "2019"]].copy()
    print(df_gdp2019)
    print(df_emission2019.describe())
    print(df_gdp2019.describe())
    
    df_2019 = pd.merge(df_emission2019, df_gdp2019, on="Country Name", how="outer")
    print(df_2019.describe())
    df_2019 = df_2019.dropna()
    df_2019.describe()
    df_2019 = df_2019.rename(columns={"2019_x":"emission", "2019_y":"gdp"})
    print(df_2019)
    
    # using scatter matrix 
    pd.plotting.scatter_matrix(df_2019, figsize=(12, 12), s=5, alpha=0.8)
    
    # correlation of variables
    df2019 = df_2019.corr()
    print(df2019)
    df_eg2019 = df_2019[["emission", "gdp"]].copy()
    
    df_norm, df_min, df_max = ct.scaler(df_eg2019)
    
    
        
        # loop over number of clusters
    for ncluster in range(2, 10):
        kmeans = cluster.KMeans(n_clusters=ncluster)
        kmeans.fit(df_eg2019)     
        labels = kmeans.labels_
        cen = kmeans.cluster_centers_
     # calculate the silhoutte score
    print(ncluster, skmet.silhouette_score(df_eg2019, labels))
    
    n = 3
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_eg2019)    
    labels = kmeans.labels_
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_eg2019["emission"], df_eg2019["gdp"], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("emission")
    plt.ylabel("gdp")
    plt.title('world cluster 2019')
    plt.savefig('cluster42019.png', dpi = 300)
    plt.show()
    
    cen = ct.backscale(cen, df_min, df_max)
    xcen = cen[:, 0]
    ycen = cen[:, 1]

    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))

    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_2019["emission"], df_2019["gdp"], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("emission")
    plt.ylabel("gdp")
    plt.title('world cluster original scale (2019) ')
    plt.savefig('cluster_orig2019.png', dpi = 300)
    plt.show()
    
    return df_emission
    
Finally = cluster(df_emission)

def fit(df_gdp):
    df_gdp2 = pd.read_csv("API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5358515.csv", skiprows=(4))
    df_poland = df_gdp2.iloc[190:191,:]
    chuka = [[2013, 0.917452], [2014, 3.914706], [2015,4.452884], [2016,2.997773],[2017,5.126897],[2018,5.945421],[2019,4.475505],[2020,-1.848395],[2021,7.277815]]
    df_poland = pd.DataFrame(data = chuka, columns = ['year', 'GDP'])
    df_poland.plot('year','GDP')
    
    def poly(x, a, b, c, d, e):
        x = x - 2010
        f = a + b*x + c*x**2 + d*x**3 + e*x**4
        return f
    param, covar = opt.curve_fit(poly, df_poland["year"], df_poland["GDP"])
    print(param)
    df_poland["fit"] = poly(df_poland["year"], *param)
    df_poland.plot("year", ["GDP", "fit"])
    plt.xlabel('year')
    plt.ylabel('GDP')
    plt.title('Graph of poland GDP against year')
    plt.savefig('fittingpoly.png', dpi = 300)
    plt.show()
    print(df_poland) 
    
    year = np.arange(1960, 2041)
    forecast = poly(year, *param)
    plt.figure()
    plt.plot(year, forecast, label="forecast")
    plt.plot(df_poland["year"], df_poland["GDP"], label="GDP")
    plt.xlabel("year")
    plt.ylabel("GDP")
    plt.title('Poland GDP for 2041 forecasted')
    plt.legend()
    plt.savefig('poly_forecast.png', dpi = 300)
    plt.show()
    
    sigma = np.sqrt(np.diag(covar))
    df_poland["fit"] = poly(df_poland["year"], *param)
    df_poland.plot("year", ["GDP", "fit"])
    plt.show()

    print("turning point", param[2], "+/-", sigma[2])
    print("GDP at turning point", param[0]/100, "+/-", sigma[0]/100)
    print("growth rate", param[1], "+/-", sigma[1])
    
    # error bandgap
    sigma = np.sqrt(np.diag(covar))
    low, up = err.err_ranges(year, poly, param, sigma)
    plt.figure()
    plt.plot(df_poland["year"], df_poland["GDP"], label="GDP")
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel("year")
    plt.ylabel("GDP")
    plt.legend()
    plt.savefig('errorband.png', dpi=300)
    plt.show()
    print(param)
    
curve_fit = fit(df_gdp)


def poland():
    import sklearn.cluster as cluster
    df_gdp2 = pd.read_csv("API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5358515.csv", skiprows=(4))
    df_poland = df_gdp2.iloc[190:191,:]
    chuka = [[2013, 0.917452], [2014, 3.914706], [2015,4.452884], [2016,2.997773],[2017,5.126897],[2018,5.945421],[2019,4.475505],[2020,-1.848395],[2021,7.277815]]
    df_poland = pd.DataFrame(data = chuka, columns = ['year', 'GDP'])
    df_poland['emission'] = 310589.9963, 303350.0061, 298299.9878, 285730.011, 289079.9866, 299799.9878, 312859.9854, 311910.0037, 295130.0049
    df_poland.describe()
    df_poland_cluster = df_poland[["GDP", "emission"]].copy()
    df_cluster, df_min, df_max = ct.scaler(df_poland_cluster)
    for ncluster in range(2, 8):
    
        kmeans = cluster.KMeans(n_clusters=ncluster)
        kmeans.fit(df_poland_cluster)     
        labels = kmeans.labels_
        cen = kmeans.cluster_centers_
        print(ncluster, skmet.silhouette_score(df_poland_cluster, labels))
        df_cluster, df_min, df_max = ct.scaler(df_poland_cluster)
    
    ncluster = 2
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_poland_cluster)     
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    xcen = cen[:, 0]
    ycen = cen[:, 1]

    plt.figure(figsize=(8.0, 8.0))

    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_poland_cluster["GDP"], df_poland_cluster["emission"], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("GDP")
    plt.ylabel("emission")
    plt.title('poland cluster')
    plt.savefig('poland_cluster.png', dpi=300)
    plt.show()


    cen = ct.backscale(cen, df_min, df_max)
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_poland["GDP"], df_poland["emission"], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("GDP")
    plt.ylabel("emission")
    plt.title('poland cluster centered to original scale')
    plt.savefig('poland_cluster_orig.png', dpi=300)
    plt.show()
    poland()