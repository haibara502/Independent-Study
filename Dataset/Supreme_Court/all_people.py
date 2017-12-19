import re
import nltk
#nltk.download()

input_file = open("supreme.conversations.txt", "r")

output_file = open("supreme.all_people.txt", "w")

iteration = 0
names = []
while 1:
	iteration += 1
#	if (iteration >= 5):
#		break;
	line = input_file.readline()
	if not line:
		break;
	contents = line.split(" +++$+++ ")
	name = contents[3]
#	print name
	names.append(name)
#	break

#print(names)
names = list(set(names))
#print(names)
names.sort()
output_file.write("\n".join(names))
print(names)
input_file.close()
output_file.close()
