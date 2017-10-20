from scipy import spatial
import re,sys
import heapq

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA

# self-defined features 
import features

# if len(sys.argv) != 3:
# 	sys.stderr.write("Usage: %s <file> <numberOfSwings>\n" % sys.argv[0])
# NOISY_FILE = sys.argv[1]
# NUMBEROfSWINGS = int(sys.argv[2])

NOISY_FILE = 'bigdatasets/40swingdataset.txt'
NUMBEROfSWINGS = 40
# it's just an estimate, if user swing 30 times, approximately 25 set maybe captured
TOP_K_NUM = NUMBEROfSWINGS - 15

SETUP			= 1
TOPOfSWING		= 2
IMPACT			= 3
FOLLOWTHROUGH	= 4
FINISH			= 5

STANDARD_FILE_NAMES = {
	SETUP: 'benchmark/setup.txt',
	TOPOfSWING: 'benchmark/topofswing.txt',
	IMPACT: 'benchmark/impact.txt',
	FOLLOWTHROUGH: 'benchmark/followthrough.txt',
	FINISH: 'benchmark/finish.txt'
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

############ extract features
X = 'X'
Y = 'Y'
Z = 'Z'

def calculateFeatures(features_list):
	output = []
	# getDistance(joints[AnkleLeft],joints[AnkleRight])*100
	joint1 = {X:features_list[features.AnkleLeft*3], Y:features_list[features.AnkleLeft*3 + 1], Z:features_list[features.AnkleLeft*3 + 2]}
	joint2 = {X:features_list[features.AnkleRight*3], Y:features_list[features.AnkleRight*3 + 1], Z:features_list[features.AnkleRight*3 + 2]}
	output.append(features.getDistance(joint1, joint2) * 100)
	#  getDistance(joints[ElbowLeft],joints[ElbowRight])*100
	joint1 = {X:features_list[features.ElbowLeft*3], Y:features_list[features.ElbowLeft*3 + 1], Z:features_list[features.ElbowLeft*3 + 2]}
	joint2 = {X:features_list[features.ElbowRight*3], Y:features_list[features.ElbowRight*3 + 1], Z:features_list[features.ElbowRight*3 + 2]}
	output.append(features.getDistance(joint1, joint2) * 100)
	# getAngle(joints[Head],joints[Neck],joints[SpineBase])
	joint1 = {X:features_list[features.Head*3], Y:features_list[features.Head*3 + 1], Z:features_list[features.Head*3 + 2]}
	joint2 = {X:features_list[features.Neck*3], Y:features_list[features.Neck*3 + 1], Z:features_list[features.Neck*3 + 2]}
	joint3 = {X:features_list[features.SpineBase*3], Y:features_list[features.SpineBase*3 + 1], Z:features_list[features.SpineBase*3 + 2]}
	output.append(features.getAngle(joint1, joint2, joint3))
	# getAngle(joints[ShoulderLeft],joints[ElbowLeft],joints[WristLeft])
	joint1 = {X:features_list[features.ShoulderLeft*3], Y:features_list[features.ShoulderLeft*3 + 1], Z:features_list[features.ShoulderLeft*3 + 2]}
	joint2 = {X:features_list[features.ElbowLeft*3], Y:features_list[features.ElbowLeft*3 + 1], Z:features_list[features.ElbowLeft*3 + 2]}
	joint3 = {X:features_list[features.WristLeft*3], Y:features_list[features.WristLeft*3 + 1], Z:features_list[features.WristLeft*3 + 2]}
	output.append(features.getAngle(joint1, joint2, joint3)/4)
	# getAngle(joints[SpineShoulder],joints[SpineBase],joints[AnkleRight])
	joint1 = {X:features_list[features.SpineShoulder*3], Y:features_list[features.SpineShoulder*3 + 1], Z:features_list[features.SpineShoulder*3 + 2]}
	joint2 = {X:features_list[features.SpineBase*3], Y:features_list[features.SpineBase*3 + 1], Z:features_list[features.SpineBase*3 + 2]}
	joint3 = {X:features_list[features.AnkleRight*3], Y:features_list[features.AnkleRight*3 + 1], Z:features_list[features.AnkleRight*3 + 2]}
	output.append(features.getAngle(joint1, joint2, joint3))
	# getKneeFlexion(joints[HipRight],joints[KneeRight],joints[AnkleRight])
	joint1 = {X:features_list[features.HipRight*3], Y:features_list[features.HipRight*3 + 1], Z:features_list[features.HipRight*3 + 2]}
	joint2 = {X:features_list[features.KneeRight*3], Y:features_list[features.KneeRight*3 + 1], Z:features_list[features.KneeRight*3 + 2]}
	joint3 = {X:features_list[features.AnkleRight*3], Y:features_list[features.AnkleRight*3 + 1], Z:features_list[features.AnkleRight*3 + 2]}
	output.append(features.getKneeFlexion(joint1, joint2, joint3)/4)
	# getKneeFlexion(joints[HipLeft],joints[KneeLeft],joints[AnkleLeft])
	joint1 = {X:features_list[features.HipLeft*3], Y:features_list[features.HipLeft*3 + 1], Z:features_list[features.HipLeft*3 + 2]}
	joint2 = {X:features_list[features.KneeLeft*3], Y:features_list[features.KneeLeft*3 + 1], Z:features_list[features.KneeLeft*3 + 2]}
	joint3 = {X:features_list[features.AnkleLeft*3], Y:features_list[features.AnkleLeft*3 + 1], Z:features_list[features.AnkleLeft*3 + 2]}
	output.append(features.getKneeFlexion(joint1, joint2, joint3)/4)

	return output

clean_data_features = {}
for i in range(SETUP, FINISH + 1):
	clean_data_features[i] = []

for i in range(SETUP, FINISH + 1):
	for j in range(TOP_K_NUM):
		clean_data_features[i].append(calculateFeatures(clean_data[i][j]))


##### Hierarchical clustering ########
clean_data_lable = {}
for i in range(SETUP, FINISH + 1): 
	ward = AgglomerativeClustering(n_clusters=2, linkage='complete',
		affinity='cosine').fit(clean_data_features[i])
	clean_data_lable[i] = ward.labels_ # it is the classification obtained by agglomerative clustering


print(clean_data_lable)

############ SVM model
# from sklearn import svm

# SVMs = {}

# SVM_C_value = {
# 	SETUP: 1,
# 	TOPOfSWING: 0.3,
# 	IMPACT: 10,
# 	FOLLOWTHROUGH: 1,
# 	FINISH:1
# }


# for i in range(SETUP, FINISH + 1):
# 	clf = svm.SVC(C = SVM_C_value[i]) 
# 	clf.fit(clean_data_features[i], clean_data_lable[i])
# 	SVMs[i] = clf

########## DecisionTree
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

DTs = {}

DT_max_depth = {
	SETUP: None,
	TOPOfSWING: 5,
	IMPACT: None,
	FOLLOWTHROUGH: None,
	FINISH: None
}

for i in range(SETUP, FINISH + 1):
	clf = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_split=2, random_state=0)
	clf.fit(clean_data_features[i], clean_data_lable[i])
	DTs[i] = clf

########### Save the trained model
from sklearn.externals import joblib
for i in range(SETUP, FINISH + 1):
	joblib.dump(DTs[i], 'user_test/classifier_%d.pkl' % i) 
