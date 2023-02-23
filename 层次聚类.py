from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from itertools import cycle
centers = [[1,1],[-1,-1],[1,-1]] #产生随机数据的中心
n_sample = 3000 #产生数据的个数
X,lables_true = make_blobs(n_samples=n_sample,
                           centers=centers,
                           cluster_std=0.6,
                           random_state=0,n_features=4) #产生数据
linkages = ['ward', 'average','complete']  #设置分层聚类函数
n_clusters = 3
ac = AgglomerativeClustering(linkage=linkages[2], n_clusters=n_clusters)
ac.fit(X)
labels = ac.labels_
plt.figure(1)
plt.clf()
colors = cycle('bgrcmykbgrmykbgrcmykbgrcmyk') #这一串东西放进迭代器中，目的是任意三个字母都能代表不同颜色

"""下面的for循环实现的功能就是plot三次，将三个蔟依次plot出来"""
for k,col in zip(range(n_clusters),colors):
    my_members = labels == k   #根据labels中的值是否等于k，重新组成一个True，False数组
    # X[my_members, 0]：取出my_members对应位置为True对应位置为True的值和坐标
    plt.plot(X[my_members, 0], X[my_members,1], col + '.')
plt.show()

