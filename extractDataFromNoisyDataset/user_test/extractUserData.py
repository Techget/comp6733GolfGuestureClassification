from scipy import spatial
import re,sys
import heapq

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA

import os

import cosDisFeatureSVM


if len(sys.argv) != 2:
	sys.stderr.write("Usage: %s <file>\n" % sys.argv[0])
USER_FILE = sys.argv[1]
# NUMBEROfSWINGS = int(sys.argv[2])

# NOISY_FILE = '30swingdataset.txt'
# NUMBEROfSWINGS = 30
# it's just an estimate, if user swing 30 times, approximately 25 set maybe captured
# TOP_K_NUM = NUMBEROfSWINGS - 5

SETUP			= 1
TOPOfSWING		= 2
IMPACT			= 3
FOLLOWTHROUGH	= 4
FINISH			= 5

# more loose threshold when compare to the one we used to extract entries
THRESHOLDS = {
	SETUP: 0.999913,
	TOPOfSWING: 0.99957,
	IMPACT: 0.9998,
	FOLLOWTHROUGH: 0.99905,
	FINISH: 0.9992
}

directory = os.path.dirname(__file__)
# print(directory)

# /import/adams/2/z5089812/comp6733GolfGuestureClassification/extractDataFromNoisyDataset/benchmark/setup.txt

STANDARD_FILE_NAMES = {
	SETUP: os.path.join(directory, "../benchmark/setup.txt"),
	TOPOfSWING: os.path.join(directory, "../benchmark/topofswing.txt"),
	IMPACT: os.path.join(directory, "../benchmark/impact.txt"),
	FOLLOWTHROUGH: os.path.join(directory, "../benchmark/followthrough.txt"),
	FINISH: os.path.join(directory, "../benchmark/finish.txt")
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
with open(USER_FILE) as f:
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

# it's like take an image from a video, those 5 images correspond to 5 phase
for i in range(SETUP, FINISH + 1):
	entry = heapq.heappop(clean_data_temp[i])
	if THRESHOLDS[i] < 1 - entry[0]:
		clean_data[i].append(entry[1])
	else:
		clean_data[i].append([0] * len(entry[1]))



# print(len(clean_data[1][0]))
#########
# each phase contains 1 samples, and each sample contains 75 feature
#########

############ extract features
clean_data_features = {}
for i in range(SETUP, FINISH + 1):
	clean_data_features[i] = []

for i in range(SETUP, FINISH + 1):
	if clean_data[i][0] == [0] * len(clean_data[i][0]):
		clean_data_features[i].append('Not Found')
	else:
		clean_data_features[i].append(cosDisFeatureSVM.calculateFeatures(clean_data[i][0]))

# print(clean_data_features)

########### load the trained model
from sklearn.externals import joblib

SVM_models = {}

for i in range(SETUP, FINISH + 1):
	SVM_models[i] = joblib.load('classifier_%d.pkl' % i) 


for i in range(SETUP, FINISH + 1):
	# print(clean_data[i])
	# print(clean_data_features[i])
	if clean_data_features[i][0] == 'Not Found':
		print('Not captured')
	else:
		print(SVM_models[i].predict(clean_data_features[i]))

