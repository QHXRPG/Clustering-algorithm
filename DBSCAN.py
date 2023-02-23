import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets

x1,y1 = datasets.make_circles(n_samples=1000, factor=.6, noise=.05)
x2,y2 = datasets.make_blobs(n_samples=300, n_features=2, centers=[[1.2,1.2]],
                            cluster_std=[[.1]], random_state=9)
X = np.concatenate((x1,x2))
plt.scatter(X[:,0],X[:,1])
plt.show()

""" 输入点，eps，点的位置j,   输出该点的集群 """
def make_cluster(x:np.ndarray,j:int, eps:float)-> list:
    k = [] #该点的集群
    for i in range(len(x)):
        dis =np.sqrt(np.sum(np.square(x[j]-x[i])))
        if dis < eps:
            k.append(x[i])
    return k

"""遍历该点的集群内的每个元素"""
def traverse_cluster(k:list, x:np.ndarray)->list:
    index=[]
    for i in range(len(x)):
        for j in range(1,len(k)):
            if k[j][0]==x[i][0] and k[j][1]==x[i][1]:
                index.append(i)
    return index

"""合并一个index内所有点的集群为一个蔟"""
def merge_cluster(k:list, index:list, x:np.ndarray,eps:float,minpts:int)->np.ndarray:
    k=pd.DataFrame(k)
    for i in range(len(index)):
        k1 = make_cluster(x,index[i],eps)
        if len(k1) >= minpts:
            k1 = pd.DataFrame(k1)
            k = pd.merge(k,k1,how='outer')  #并集方式合并数据
    return k.values

"""DBSCAN算法第一步"""
def step_1(x:np.ndarray, j:int, eps:float,minpts:int)->(np.ndarray,list):
    return merge_cluster(k=make_cluster(x,j,eps),
                         index=traverse_cluster(make_cluster(x,j,eps),x),
                         x=x,
                         eps=eps,
                         minpts=minpts),\
           traverse_cluster(make_cluster(x,j,eps),x)

""" DBSCAN算法 """
def DBSCAN(x:np.ndarray, eps:float, minpts:int):
    labels =[1 for i in range(len(x))]
    index=[]
    for i in range(len(x)):
        if i not in index:
            if len(make_cluster(x,i,eps)) >= minpts:
                cluster,index_i = step_1(x,i,eps,minpts)
                for j in index_i:
                    labels[j] = i
                    index.append(j)
    return labels