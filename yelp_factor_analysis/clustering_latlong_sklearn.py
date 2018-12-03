import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter

outFile = open('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset_csv/yelp__business_combined.csv','r')
final_dataset = pd.read_csv(outFile)
#final_dataset = final_dataset.iloc[0:100]
final_dataset.fillna(final_dataset.mean(), inplace=True)
feature_dataset = final_dataset[['latitude','longitude']]

id_n = 20
kmeans = KMeans(n_clusters=id_n, random_state=0).fit(feature_dataset)
id_label=kmeans.labels_

ptsymb = ['#800000','#D2691E','#DAA520','#F4A460','#BC8F8F','#DEB887','#FFDEAD','#FFF8DC','#000000','#2F4F4F','#778899','#808080','#D3D3D3','#FFF0F5','#FFFFF0','#F0FFFF','#F0FFF0','#FF1493','#4B0082','#8B008B','#E6E6FA','#7CFC00','#FFFF00','#FF4500']
plt.figure(figsize=(12,12))
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
clusters = {}
busines_id = np.array([])
cluster_id = np.array([])

for i in range(id_n):
    cluster=np.where(id_label==i)[0]
    clusters[i] = {}
    clusters[i]['latitude'] = list(feature_dataset.latitude[cluster].values)
    clusters[i]['longitude'] = list(feature_dataset.longitude[cluster].values)
    clusters[i]['business_id'] = list(final_dataset.business_id[cluster].values)
    clusters[i]['cluster'] = [i]*(len(clusters[i]['business_id']))
    busines_id = np.append(busines_id,clusters[i]['business_id'],axis=0)
    cluster_id = np.append(cluster_id, clusters[i]['cluster'], axis =0)
    plt.scatter(clusters[i]['latitude'],clusters[i]['longitude'],c=ptsymb[i])

location_df = pd.DataFrame()
location_df['business_id'] = busines_id
location_df['cluster_location'] = cluster_id
location_df.to_csv('/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp-business-trends/yelp_dataset_csv/yelp__business_location_20.csv')

plt.savefig('kmeans.jpg')
plt.show()