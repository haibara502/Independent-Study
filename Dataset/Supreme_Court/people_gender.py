import re
#import nltk
#nltk.download()

input_file = open("supreme.gender.txt", "r")

output_file = open("supreme.people_gender.txt", "w")

iteration = 0
gender = []
while 1:
	iteration += 1
	line = input_file.readline()
	if not line:
		break;
	contents = line.split(" +++$+++ ")
	if (contents[-1] == "male\n"):
		contents[-1] = "0"
	else :
		if (contents[-1] == "female\n"):
			contents[-1] = "1"
		else:
			if (contents[-1] == "NA\n"):
				contents[-1] = "-1"
			else:
				print("." + contents[-1] + '.')
				exit()
	gender.append(contents[0:2])
#break
final_gender = []
for item in gender:
	combination = " ".join(item)
	final_gender.append(combination)


gender = list(set(final_gender))
print(gender)
gender.sort()

print(gender)


output_file.write("\n".join(gender))
input_file.close()
output_file.close()
