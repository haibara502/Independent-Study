import re
import nltk
#Already downloaded
#nltk.download() 
import numpy as np

#The input file to read the conversations.
convers_file = open("Original/supreme.conversations.txt", "r")
#The input file to read the genders.
gender_file = open("Original/supreme.gender.txt", "r")
#The input file to read the names.
#name_file = open()
#The output file to store the dataset input
gender_train_output_x = open("Last_Utterance/Gender/supreme_gender_train_input.txt", "w")
#The output file to store the dadtaset output
gender_train_output_y = open("Last_Utterance/Gender/supreme_gender_train_output.txt", "w")
gender_test_output_x = open("Last_Utterance/Gender/supreme_gender_test_input.txt", "w")
gender_test_output_y = open("Last_Utterance/Gender/supreme_gender_test_output.txt", "w")

judge_train_output_x = open("Last_Utterance/Judge/supreme_judge_train_input.txt", "w")
judge_train_output_y = open("Last_Utterance/Judge/supreme_judge_train_output.txt", "w")
judge_test_output_x = open("Last_Utterance/Judge/supreme_judge_test_input.txt", "w")
judge_test_output_y = open("Last_Utterance/Judge/supreme_judge_test_output.txt", "w")

charact_train_output_x = open("Last_Utterance/Character/supreme_charac_train_input.txt", "w")
charact_train_output_y = open("Last_Utterance/Character/supreme_charac_train_output.txt", "w")
charact_test_output_x = open("Last_Utterance/Character/supreme_charac_test_input.txt", "w")
charact_test_output_y = open("Last_Utterance/Character/supreme_charac_test_output.txt", "w")

#Read all the people genders, create a dictionary to store the info
people_info = dict()
while 1:
	line = gender_file.readline()
	if not line:
		break
	info = line.split(' +++$+++ ')
	label = 2
	if info[1] == 'female\n':
		label = 1
	else :
		if info[1] == 'male\n':
			label = 0
	people_info[info[0]] = label

#Start to read the conversations and create the dataset
utterances = []
speakers = []
genders = []
total = 0

cache_gender_output_x = []
cache_gender_output_y = []

cache_judge_output_x = []
cache_judge_output_y = []

cache_charact_output_x = []
cache_charact_output_y = []
while 1:
	line = convers_file.readline()
	if not line:
		break;
	contents = line.split(" +++$+++ ", 14)

	total = total + 1
	print total

	#If this utterance is the beginning of the conversation
	new_conversation = contents[2]
	if new_conversation == "FALSE":
		utterances = []
		speakers = []
		genders = []
		valid = 1

	content = contents[-1]
	wordtokens = nltk.tokenize.word_tokenize(content)
	final = ' '.join(wordtokens)
	utterances.append(final)

	speaker = contents[3]
	speakers.append(speaker)
	print "speaker"
	print speaker

	gender = people_info[speaker]
	genders.append(gender)
	print "gender: "
	print gender

	justice_condition = contents[4]
	is_justice = 0
	if justice_condition == "JUSTICE":
		is_justice = 1

	character = is_justice
	if character == 0:
		character = 2
		repre_side = contents[6]
		if repre_side == "PETITIONER":
			character = 1
		print repre_side
		print character

	total_utterance = len(utterances)
	print "total_utterance:"
	print total_utterance

	if (gender > 1):
		continue
	if new_conversation == "FALSE":
		continue

	cache_gender_output_x.append(speakers[-2] + '\n')
	cache_gender_output_x.append(utterances[-2] + '\n')
	cache_gender_output_y.append(str(gender) + '\n')

#cache_judge_output_x.append(speakers[-2] + '\n')
#cache_judge_output_x.append(utterances[-2] + '\n')
	cache_judge_output_y.append(str(is_justice) + '\n')

#cache_charact_output_x.append(speakers[-2] + '\n')
#cache_charact_output_x.append(utterances[-2] + '\n')
	cache_charact_output_y.append(str(character) + '\n')

print "Process ends."
print "Start to shuffle."

indices = np.arange(len(cache_gender_output_y))
np.random.shuffle(indices)

print indices
print "Shuffle ends."

print len(cache_gender_output_x)
print cache_gender_output_x

samples = len(cache_gender_output_y)
number_of_train = int(samples * 0.9)

for i in range(number_of_train):
	index = indices[i]

	gender_train_output_x.write(cache_gender_output_x[2 * index])
	gender_train_output_x.write(cache_gender_output_x[2 * index + 1])
	gender_train_output_y.write(cache_gender_output_y[index])

	judge_train_output_x.write(cache_gender_output_x[2 * index])
	judge_train_output_x.write(cache_gender_output_x[2 * index + 1])
	judge_train_output_y.write(cache_judge_output_y[index])

	charact_train_output_x.write(cache_gender_output_x[2 * index])
	charact_train_output_x.write(cache_gender_output_x[2 * index + 1])
	charact_train_output_y.write(cache_charact_output_y[index])

for i in range(number_of_train, samples):
	index = indices[i]

	gender_test_output_x.write(cache_gender_output_x[2 * index])
	gender_test_output_x.write(cache_gender_output_x[2 * index + 1])
	gender_test_output_y.write(cache_gender_output_y[index])

	judge_test_output_x.write(cache_gender_output_x[2 * index])
	judge_test_output_x.write(cache_gender_output_x[2 * index + 1])
	judge_test_output_y.write(cache_judge_output_y[index])

	charact_test_output_x.write(cache_gender_output_x[2 * index])
	charact_test_output_x.write(cache_gender_output_x[2 * index + 1])
	charact_test_output_y.write(cache_charact_output_y[index])

