from scipy import spatial
import re,sys

# if len(sys.argv) != 2:
#     sys.stderr.write("Usage: %s <file>\n" % sys.argv[0])
#     sys.exit(1)
# standard_file = sys.argv[1]

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

NOISY_FILE = '10swingdataset.txt'

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


# this threshold is tunable, if you feel there should be more entries belong to setup phase, tune down the threshold
SETUP_SIMILARITY_THRESHOLD = 0.99952
TOPOfSWING_SIMILARITY_THRESHOLD = 0.998852
IMPACT_THRESHOLD = 0.99857
FOLLOWTHROUGH_THRESHOLD = 0.9979
FINISH_THRESHOLD = 0.9984

clean_data = {}
clean_data_frame = {}

for i in range(1,6):
	clean_data[i] = []
	clean_data_frame[i] = []
	
for frameNum,entry in enumerate(noisy_data):

	for i in range(1,6):
		cos_similarity = 1 - spatial.distance.cosine(standard_data[i], entry)
	
		if i == SETUP and cos_similarity > SETUP_SIMILARITY_THRESHOLD:
			clean_data[i].append(entry)
			clean_data_frame[i].append(frameNum)
			break
		
		if i == TOPOfSWING and cos_similarity > TOPOfSWING_SIMILARITY_THRESHOLD:
			clean_data[i].append(entry)
			clean_data_frame[i].append(frameNum)
			break

		if i == IMPACT and cos_similarity > IMPACT_THRESHOLD:
			clean_data[i].append(entry)
			clean_data_frame[i].append(frameNum)
			break

		if i == FOLLOWTHROUGH and cos_similarity > FOLLOWTHROUGH_THRESHOLD:
			clean_data[i].append(entry)
			clean_data_frame[i].append(frameNum)
			break

		if i == FINISH and cos_similarity > FINISH_THRESHOLD:
			clean_data[i].append(entry)
			clean_data_frame[i].append(frameNum)
			break

#print(len(noisy_data))
# print(clean_data)
#print(clean_data[-1])
#print('#####')
#print(standard_data)
#print (str(len(noisy_data)) + " frames captured")
#print (str(len(noisy_data)/30) + " seconds")
# for i in range(1,6):
#	print(clean_data[i][0])
#print(clean_data[SETUP][1])

print (len(clean_data[FOLLOWTHROUGH]))