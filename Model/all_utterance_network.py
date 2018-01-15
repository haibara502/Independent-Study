import numpy as np
import keras
from scipy.sparse import csr_matrix
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers import Bidirectional, LSTM, Concatenate
from keras.models import Model
from keras.optimizers import SGD
import string
from keras import backend as K

WORD_VECTOR_FILE = '../Dataset/Supreme_Court/vector.txt'
WORD_EMBEDDING_DIM = 100
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
NUMBER_OF_UTTERANCE = 50
SENTENCE_LENGTH = 900
NUM_HIDDEN = 50
NUMBER_OF_NAMES = 327
NAME_EMBEDDING_SIZE = 100
LABEL_SIZE = 2
UTTERANCE_LENGTH = 50
WORD_NUMBER = 10787
VOCABULARY_SIZE = 10788

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

word_vector_matrix = np.zeros((VOCABULARY_SIZE, WORD_EMBEDDING_DIM))
for name, index in word_ids.items():
	this_vector = word_vector[name]
	word_vector_matrix[index] = this_vector

print('Start to read the text.')

TEXT_TRAIN_INPUT_FILE = '../Dataset/Supreme_Court/supreme_last_utterance.input'
TEXT_TEST_INPUT_FILE = ''
LABEL_TRAIN_INPUT_FILE = '../Dataset/Supreme_Court/supreme_last_utterance.output'
LABEL_TEST_INPUT_FILE = ''

def generate_from_file(paths):
	while 1:
		text_input_file = open(paths[0], "r")
		label_input_file = open(paths[1], "r")

		MAX_WORDS = 10788
		MAX_NAME = 326
		all_names = dict()

		while 1:
			name = text_input_file.readline()
			if not name:
				break
			number = int(text_input_file.readline())
			utterances = []
			for i in range(number):
				utterance = text_input_file.readline()
				words = utterance.split()
				sentence = []
				for item in words:
					index = word_ids.get(item)
					if index is not None:
						sentence.append(index)
					else:
						sentence.append(VOCABULARY_SIZE - 1)
				while len(sentence) < SENTENCE_LENGTH:
					sentence.append(VOCABULARY_SIZE - 1)
				if len(sentence) > SENTENCE_LENGTH:
					sentence = sentence[:SENTENCE_LENGTH]
				utterances.append(sentence)

#scripts.append(utterances)

			name = name[:-1]
			name_id = all_names.get(name)
			if name_id is None:
				all_names[name] = len(all_names)
				name_id = all_names.get(name)
#			names.append(np.array(name_id))
	
			label = int(label_input_file.readline())
#			labels.append(np.array(label))
			MAX_LABEL = 2
			label = to_categorical(label, num_classes = MAX_LABEL)

			yield ({np.array(utterances), np.array(name_id)}, {np.array(label)})

		text_input_file.close()
		label_input_file.close()


#train_data = read_data([TEXT_TRAIN_INPUT, LAEBL_TRAIN_INPUT])
#test_data = read_data([TEXT_TEST_INPUT, LABEL_TEST_INPUT])

#scripts_train = train_data[0]
#names_train = train_data[1]
#labels_train = train_data[2]

#scripts_test = test_data[0]
#names_test = test_data[1]
#labels_test = test_data[2]

#std_output = open("std.txt", "w")
#for item in labels_test:
#	std_output.write(str(item) + '\n')
"""
print("Shuffle the data")

print len(scripts)
print len(names)
print len(labels)

scripts = np.array(scripts)
indices = np.arange(scripts.shape[0])
np.random.shuffle(indices)

#Shuffle the scripts
scripts = np.array(scripts)
scripts = scripts[indices]

#Shuffle the names
names = np.array(names)
names = names[indices]

#Shuffle the labels
labels = np.array(labels)
labels = labels[indices]

print "label shape"
print labels.shape
print "scripts shape"
print scripts.shape
print "names shpae"
print names.shape

num_validation_samples = int(VALIDATION_SPLIT * scripts.shape[0])

scripts_train = scripts[:-num_validation_samples]
scripts_valid = scripts[-num_validation_samples:]
print "scripts_trani"
print scripts_train.shape
scripts_valid = scripts[-num_validation_samples:]
names_train = names[:-num_validation_samples]
print "names_trani"
print names_train.shape
names_valid = names[-num_validation_samples:]
labels_train = labels[:-num_validation_samples]
print "labels_train"
print labels_train.shape
labels_valid = labels[-num_validation_samples:]

print 'script shape'
print scripts.shape
"""

script_input = Input(shape=(NUMBER_OF_UTTERANCE, SENTENCE_LENGTH,))
script_embedding = Embedding(VOCABULARY_SIZE, output_dim = EMBEDDING_DIM, weights = [word_vector_matrix])(script_input) 
script_output = TimeDistributed(Bidirectional(LSTM(NUM_HIDDEN), merge_mode='concat'))(script_embedding)

model = Model(inputs=script_input, outputs=script_output)

print(model.input_shape)
print(model.output_shape)

name_input = Input(shape=(NUMBER_OF_UTTERANCE, ))
name_output = Embedding(NUMBER_OF_NAMES, output_dim = EMBEDDING_DIM)(name_input)
#name_output = Flatten()(name_output)

model = Model(inputs=name_input, outputs=name_output)
print(model.input_shape)
print(model.output_shape)
model.summary()

utterance_input = Concatenate(axis = -1)([script_output, name_output])

#output = Dense(NUM_HIDDEN, activation ='sigmoid')(utterance_input)
output = Dense(LABEL_SIZE, activation='softmax')(utterance_input)
print output.shape

model = Model(inputs = [script_input, name_input], outputs = output)
print(model.input_shape)
print(model.output_shape)
model.summary()

model.compile(optimizer = SGD(lr = 0.1), loss='binary_crossentropy', metrics=['accuracy'])

#print scripts_train.shape
#print names_train.shape
#print labels_train.shape

model.fit_generator(generate_from_file([TRAIN_INPUT_FILE, TRAIN_OUTPUT_FILE]), 989, epochs = 2)
#score, acc = model.evaluate([scripts_valid, names_valid], labels_valid, batch_size = 20)

OUTPUT_FILE = open("output.txt", "w")
prediction = model.predict_generator(generate_from_file([TEST_INPUT_FILE, TEST_OUTPUT_FILE]), val_samples = 5055, verbose = 1)
for item in prediction:
	if item[0] > item[1]:
		OUTPUT_FILE.write("0\n")
	else:
		OUTPUT_FILE.write("1\n")
#OUTPUT_FILE.write(np.argmax(item))

#print('Test score:', score)
#print('Test accuracy:', acc)
