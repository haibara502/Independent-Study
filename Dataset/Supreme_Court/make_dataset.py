import re
import nltk
#Already downloaded
#nltk.download() 

#The input file to read the conversations.
convers_file = open("supreme.conversations.txt", "r")
#The input file to read the genders.
gender_file = open("supreme.gender.txt", "r")
#The input file to read the names.
#name_file = open()
#The output file to store the dataset input
output_x = open("supreme_input.txt", "w")
#The output file to store the dadtaset output
output_y = open("supreme_output.txt", "w")

#Read all the people genders, create a dictionary to store the info
people_info = dict()
while 1:
	line = gender_file.readline()
	if not line:
		break
	info = line.split(' +++$+++ ')
	label = -1
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
valid = 0
while 1:
	line = convers_file.readline()
	if not line:
		break;
	contents = line.split(" +++$+++ ", 14)
#	print contents
	valid = valid + 1

	#If this utterance is the beginning of the conversation
	new_conversation = contents[2]
	if new_conversation == "FALSE":
		utterances = []
		speakers = []
		genders = []
		valid = 1
	print "new_conversation: "
	print new_conversation
#	print "utterances:"
#print utterances

	content = contents[-1]
	wordtokens = nltk.tokenize.word_tokenize(content)
	final = ' '.join(wordtokens)
	utterances.append(final)
#print final

	speaker = contents[3]
	speakers.append(speaker)
	print "speaker"
	print speaker

	gender = people_info[speaker]
	genders.append(gender)
	print "gender: "
	print gender
	if (gender < 0):
		valid = valid - 1

	total_utterance = len(utterances)
	print "total_utterance:"
	print total_utterance
#	print "utterances: "
#	print utterances
	print 'valid'
	print valid
	if valid <= 1:
		continue

	output_x.write(str(valid - 1) + '\n')
	for i in range(1, total_utterance):
		if genders[i] == -1:
			continue
		output_x.write(speakers[i - 1] + '\n')
		output_x.write(utterances[i - 1] + '\n')
		output_y.write(str(genders[i]) + '\n')
