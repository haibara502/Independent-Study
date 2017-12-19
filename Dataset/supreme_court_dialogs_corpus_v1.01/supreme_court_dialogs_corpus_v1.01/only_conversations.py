import re
import nltk
nltk.download()

input_file = open("supreme.conversations.txt", "r")

output_file = open("supreme.only.conversations.txt", "w")

iteration = 0
while 1:
	iteration += 1
	line = input_file.readline()
	if not line:
		break;
	contents = line.split(" ", 14)
#	print contents
	new_conversation = contents[4]
	print new_conversation
	if new_conversation == "FALSE":
		output_file.write("#\n")
	content = contents[-1]
	while content.find('+++$+++ ') >= 0:
		print content.find('+++$+++ ')
		content = content[content.index('+++$+++ ') + 8:]
#print content
	wordtokens = nltk.tokenize.word_tokenize(content)
#	print wordtokens
	final = ' '.join(wordtokens)
	final += '\n'
	print final
	output_file.write(final)

input_file.close()
output_file.close()
