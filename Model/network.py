import numpy as np
from scipy.sparse import csr_matrix
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.layers import Dense, Input
from keras.layers import Embedding
from keras.models import Model
import string

WORD_VECTOR_FILE = '../Dataset/Supreme_Court/vector.txt'
EMBEDDING_DIM = 100

print('Load pretrained word vectors.')

word_vector_file = open(WORD_VECTOR_FILE, "r")

texts = []
word_vector = dict()
word_ids = dict()
while 1:
	line = word_vector_file.readline()
	if not line:
		break
	line = line.split()
	word = line[0]
	vector = np.asarray(line[1:], dtype='float32')
	word_vector[word] = vector
	texts.append(word)

	word_index = word_ids.get(word)
	if word_index is None:
		word_ids[word] = len(word_ids)

word_vector_file.close()

print('Start to read the text.')

TEXT_INPUT_FILE = '../Dataset/Supreme_Court/supreme_input.txt'
LABEL_INPUT_FILE = '../Dataset/Supreme_Court/supreme_output.txt'
text_input_file = open(TEXT_INPUT_FILE, "r")
label_input_file = open(LABEL_INPUT_FILE, "r")

labels = []
names = []
scripts = []

MAX_WORDS = 10788
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)

MAX_NAME = 311
name_tokenizer = Tokenizer(num_words = MAX_NAME)

all_names = dict()

while 1:
	num_utters = text_input_file.readline()
	if not num_utters:
		break
	num_utters = int(num_utters)
	print "Num_utters"
	print num_utters

	utterances = []
	name_one_script = []
	label = []
	for i in range(num_utters):
		name = text_input_file.readline()
		utterance = text_input_file.readline()
		words = utterance.split()
		sentence = []
		for item in words:
			index = word_vector.get(item)
			if index is not None:
				sentence.append(index)
			else:
				sentence.append(np.zeros(100))
		utterances.append(sentence)

		name = name[:-1]
		name_id = all_names.get(name)
		if name_id is not None:
			name_one_script.append(name_id)
		else:
			all_names[name] = len(all_names)
			name_id = all_names.get(name)
			name_one_script.append(name_id)
		label_index = int(label_input_file.readline())
		label.append(label_index)	

	scripts.append(utterances)
	names.append(name_one_script)
	labels.append(label)

MAX_LABEL = 2

name_tokenizer.fit_on_texts(all_names)
names_temp = []
for item in names:
	temp_item = np.array(item)
	n_sample = len(temp_item)
	temp_name = np.eye(MAX_NAME + 1)[temp_item]
	names_temp.append(temp_name)
names = names_temp
print('Reading ends.')

print("Shuffle the data")

print len(scripts)
print len(names)
print len(labels)
exit()
scripts = np.array(scripts)
indices = np.arange(scripts.shape[0])
np.random.shuffle(indices)

#Shuffle the scripts
scripts_temp = []
for i in range(scripts.shape[0]):
	scripts_temp.append(scripts[indices[i]])
scripts = np.array(scripts_temp)

#Shuffle the names
names_temp = []
names = np.array(names)
for i in range(names.shape[0]):
	names_temp.append(names[indices[i]])
names = np.array(names_temp)

print scripts.shape
print names.shape
print labels.shape
exit()
labels_temp = []
labels = np.array(labels)
for i in range(labels.shape[0]):
	labels_temp.append(labels[indices[i]])
labels = np.array(labels_temp)
print labels
exit()

num_validation_samples = int(VALIDATION_SPLIT * scripts.shape[0])

scripts_train = scripts[:-num_validation_samples]
scripts_valid = scripts[-num_validation_samples:]
names_train = names[:-num_validation_samples]
names_valid = names[-num_validation_samples:]
labels_train = labels[:-num_validation_samples]
labels_valid = labels[-num_validation_samples]

script_input = Input(shape=(NUMBER_OF_UTTERANCE, SENTENCE_LENGTH, WORD_NUM))
script_embedding = TimeDistributed(
	Embedding(WORD_NUM, output_dim=WORD_EMBEDDING_SIZE, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
			  trainable=True)(script_input))
script_output = Bidirectional(LSTM(NUM_HIDDEN, return_sequences=True), merge_mode='concat')(script_embedding)

model = Model(inputs=script_input, outputs=script_output)

print(model.input_shape)
print(model.output_shape)

name_input = Input(shape=(NUMBER_OF_UTTERANCE,))
name_output = Embedding(NUMBER_OF_NAMES, output_dim=NAME_EMBEDDING_SIZE)(name_input)

model = Model(inputs=name_input, outputs=name_output)
print(model.input_shape)
print(model.output_shape)

utterance_input = Concatenate(axis=-1)([script_output, name_output])

output = LSTM(SCRIPT_EMBEDDING_SIZE)(utterance_input)
output = Dense(LABEL_SIZE, activation='softmax')(output)

model = Model(inputs=[script_input, name_input], outputs=output)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([scripts_train, names_train], labels_train, epochs=50, batch_size=1)
