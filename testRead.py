import re

fname = '1.txt'

iter_num = 0
output_list = []
temp_list = []
contents = []

with open(fname) as f:
    contents = f.readlines()

for content in contents:
    content = content.strip('\n')
    content = re.sub(r' ', '', content)
    numbers = content.split(',')
    if iter_num >= 30:
        output_list.append(temp_list)
        temp_list = []
        temp_list.extend(numbers)
        iter_num = 0
    else:
        temp_list.extend(numbers)
        iter_num += 1

# print(output_list)
# print(len(output_list))
# for entry in output_list:
#     print(output_list)

with open('30numberPerEntry.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(str(output) for output in output_list))



