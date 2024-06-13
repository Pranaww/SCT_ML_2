import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans  
data=pd.read_csv("Mall_customers.csv")
#clustering based on only income and spendig score
req=data.iloc[:,[3,4]].values
#using elbow method to find optimal number of clusters
wcss_list= []
for i in range(1,11):#assumption of maxclusters to be 11
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=42)
    kmeans.fit(req)
    wcss_list.append(kmeans.inertia_)
plt.plot(range(1,11),wcss_list)
plt.title("ELbow method")
plt.xlabel("No of clusters")
plt.ylabel("wcss_list")
plt.show()
#assuming the most suitable cluster value from graph
#trainig the model
kmeans=KMeans(n_clusters=5,init="k-means++",random_state=42)
y_predict=kmeans.fit_predict(req)
#visulaizing the clusters  
plt.scatter(req[y_predict == 0, 0], req[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster  
plt.scatter(req[y_predict == 1, 0], req[y_predict == 1, 1], s = 100, c = 'green', label = 'Cluster 2') #for second cluster  
plt.scatter(req[y_predict== 2, 0], req[y_predict == 2, 1], s = 100, c = 'red', label = 'Cluster 3') #for third cluster  
plt.scatter(req[y_predict == 3, 0], req[y_predict == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') #for fourth cluster  
plt.scatter(req[y_predict == 4, 0], req[y_predict == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') #for fifth cluster  
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')   
plt.title('Clusters of customers')  
plt.xlabel('Annual Income (k$)')  
plt.ylabel('Spending Score (1-100)')  
plt.legend()  
plt.show()  

