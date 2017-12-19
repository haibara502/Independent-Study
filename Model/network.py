import numpy as np

WORD_VECTOR_FILE = 'vector.txt'

print('Load pretrained word vectors.')

word_vector_file = fopen(WORD_VECTOR_FILE, "r")

word_vector = dict()
while 1:
	line = word_vector_file.readline()
	if not line:
		break
	line = line.split()
	word = line[0]
	vector = np.asarray(line[1:], dtype = 'float32')
	word_vector[word] = vector

word_vector_file.close()

print('Start to read the text.')

text_input_file = fopen(TEXT_INPUT_FILE, "r")
label_input_file = fopen(LABEL_INPUT_FILE, "r")

labels = []
names = []
scripts = []
while 1:
	read the number
	for i in range number:
		read name
		suoyin name
		read sentence
		text.append(sentence)
	script.append(text)

print('Reading ends.')

labels = to_categorical(np.asarray(labels))
names = to_categorical(np.asarray(names))

print("Prepare embedding matrix")

tokenizer = Tokenizer(num_words = MAX_WORDS)
word_index = tokenizer.word_index
num_words = min(MAX_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
	if i >= MAX_WORDS:
		continue
	embedding_vector.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

print('Convert name to one-hot vector')
name_tokenizer = Tokenizer(num_words = MAX_PEOPLE)
name_tokenizer.fit_on_texts(names)


