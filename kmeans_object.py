#�������ı��
#-*-coding:utf-8-*-
__author__ = 'guojunfeng'
from numpy import *
import pandas as pd
#���ø��ֺ�����
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
#�������ݵ���
class load:
    def __init__(self):
        print ("init load") # never prints
    def load_data(self):
        data, label = make_blobs (n_samples=500, n_features=2, centers=2, random_state=0, center_box=(10, 20))
        return data
#kmeans�������
class kmeans:
    def __init__(self):
        print('init kmeans')
    def kmeans_cluster(self,data):
        kmeans=KMeans(n_clusters=2)
        kmeans.fit(data)
        label_pred=kmeans.labels_
        centroids = kmeans.cluster_centers_
        inertia = kmeans.inertia_
        print('classify outcom')
        print(label_pred)
        print('classify cente')
        print(centroids)
        print('classify loss')
        print(inertia)
        return label_pred
#���ӻ�����
class matplot:
    def __init__(self):
        print('init matplot')
    def visualize(self,label_pred,data):
        mark = ['or', 'ob']
        j = 0
        for i in label_pred:
            plt.plot (data[j][0], data[j][1], mark[i], markersize=5)
            j += 1
        plt.show ()
#����load��Ķ���
l=load()
#ͨ���������load����data����
x=l.load_data()
#����kmeans����
k=kmeans()
#����kmeans��cluster����
pred_label=k.kmeans_cluster(x)
mp=matplot()
mp.visualize(pred_label,x)




