from scipy import spatial
import re,sys
import heapq

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA

# if len(sys.argv) != 3:
# 	sys.stderr.write("Usage: %s <file> <numberOfSwings>\n" % sys.argv[0])
# NOISY_FILE = sys.argv[1]
# NUMBEROfSWINGS = int(sys.argv[2])

NOISY_FILE = '30swingdataset.txt'
NUMBEROfSWINGS = 30
# it's just an estimate, if user swing 30 times, approximately 25 set maybe captured
TOP_K_NUM = NUMBEROfSWINGS - 5

SETUP			= 1
TOPOfSWING		= 2
IMPACT			= 3
FOLLOWTHROUGH	= 4
FINISH			= 5

STANDARD_FILE_NAMES = {
	SETUP: 'setup.txt',
	TOPOfSWING: 'topofswing.txt',
	IMPACT: 'impact.txt',
	FOLLOWTHROUGH: 'followthrough.txt',
	FINISH: 'finish.txt'
}

standard_data = {}
for i in range(1,6):
	temp_list = []
	with open(STANDARD_FILE_NAMES[i]) as f:
		for line in f:
			temp = line.split(',')
			temp = temp[:3] ## remove the last explanation word
			temp = [float(x) for x in temp]
			temp_list.extend(temp)
	standard_data[i] = temp_list


noisy_data = []
DATA_ENTRY_AMOUNT = 25
entry_read_count = 0
entry_read = []
with open(NOISY_FILE) as f:
	for line in f:
		if entry_read_count == DATA_ENTRY_AMOUNT:
			noisy_data.append(entry_read)
			entry_read = []
			entry_read_count = 0

		temp = line.split(',')
		temp = temp[:3]
		temp = [float(x) for x in temp]
		entry_read.extend(temp)
		entry_read_count += 1

# constains heapq
clean_data_temp = {}
clean_data = {}

# initialization
for i in range(SETUP, FINISH + 1):
	clean_data[i] = []
	clean_data_temp[i] = []

# add to heapq, which is a min-heap, the similarity make corresponding change,
# does not include `1 -`. 
for entry in noisy_data:
	for i in range(SETUP, FINISH + 1):
		cos_similarity_without_minus_1 = spatial.distance.cosine(standard_data[i], entry)
		heapq.heappush(clean_data_temp[i], (cos_similarity_without_minus_1, entry))

# pick the top-k entry, which is closest to standard gesture for each phase
for i in range(SETUP, FINISH + 1):
	for j in range(TOP_K_NUM):
		clean_data[i].append(heapq.heappop(clean_data_temp[i])[1])

# print(clean_data[1][0])

#########
# each phase contains TOP_K_NUM samples, and each sample contains 75 feature
#########

##### PCA #####
def doPCA(data):
	pca = PCA(n_components=3)
	return pca.fit_transform(data)

clean_data_pca = {}
for i in range(SETUP, FINISH + 1):
	clean_data_pca[i] = doPCA(clean_data[i])

# print(clean_data_pca[SETUP][0])

##### Hierarchical clustering ########
clean_data_lable = {}
for i in range(SETUP, FINISH + 1): 
	ward = AgglomerativeClustering(n_clusters=2, linkage='complete',
		affinity='cosine').fit(clean_data_pca[i])
	clean_data_lable[i] = ward.labels_ # it is the classification obtained by agglomerative clustering

##### print out 3d image, if n_components = 3
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for i in range(0, TOP_K_NUM):
# 	xs = [clean_data_pca[SETUP][i][0]]
# 	ys = [clean_data_pca[SETUP][i][1]]
# 	zs = [clean_data_pca[SETUP][i][2]]
# 	if label[i] == 0:
# 		ax.scatter(xs, ys, zs, c=10, marker='o')
# 	elif label[i] ==1:
# 		ax.scatter(xs, ys, zs, c=50, marker='^')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
######

############ SVM model
from sklearn import svm

SVMs = {}
for i in range(SETUP, FINISH + 1):
	clf = svm.SVC()
	clf.fit(clean_data_pca[i], clean_data_lable[i])
	SVMs[i] = clf


########### Save the trained model
from sklearn.externals import joblib
for i in range(SETUP, FINISH + 1):
	joblib.dump(SVMs[i], 'classifier_%d.pkl' % i) 
