from scipy import spatial
import re

# dataSetI = [3, 45, 7, 2]
# dataSetII = [2, 54, 13, 15]
# result = 1 - spatial.distance.cosine(dataSetI, dataSetII) # subtract from 1 to get the similarity
# print(result)

standard_file = 'setup.txt'
noisy_file = 'dataset.txt'

standard_data = []
with open(standard_file) as f:
	for line in f:
		temp = line.split(',')
		temp = temp[:3]
		temp = [float(x) * 10 for x in temp]
		standard_data.extend(temp)

noisy_data = []
DATA_ENTRY_AMOUNT = 25
entry_read_count = 0
entry_read = []
with open(noisy_file) as f:
	for line in f:
		if entry_read_count == DATA_ENTRY_AMOUNT:
			noisy_data.append(entry_read)
			entry_read = []
			entry_read_count = 0

		temp = line.split(',')
		temp = temp[:3]
		temp = [float(x) * 10 for x in temp]
		entry_read.extend(temp)
		entry_read_count += 1


# this threshold is tunable, if you feel there should be more entries belong to setup phase, tune down the threshold
SIMILARITY_THRESHOLD = 0.9994
clean_data = []
for entry in noisy_data:
	cos_similarity = 1 - spatial.distance.cosine(standard_data, entry)
	# print(cos_similarity)
	if cos_similarity > SIMILARITY_THRESHOLD:
		clean_data.append(entry)

print(len(noisy_data))
print(len(clean_data))
print(clean_data[-1])
print('#####')
print(standard_data)


