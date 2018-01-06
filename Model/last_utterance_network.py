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

TEXT_INPUT_FILE = '../Dataset/Supreme_Court/supreme_last_utterance.input'
LABEL_INPUT_FILE = '../Dataset/Supreme_Court/supreme_last_utterance.output'
text_input_file = open(TEXT_INPUT_FILE, "r")
label_input_file = open(LABEL_INPUT_FILE, "r")

labels = []
names = []
scripts = []

MAX_WORDS = 10788
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)

MAX_NAME = 326

all_names = dict()

while 1:
	name = text_input_file.readline()
	if not name:
		break
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
	if len(sentence) <= SENTENCE_LENGTH :
		scripts.append(sentence)

	name = name[:-1]
	name_id = all_names.get(name)
	if name_id is None:
		all_names[name] = len(all_names)
		name_id = all_names.get(name)
	if len(sentence) <= SENTENCE_LENGTH:
		names.append(np.array(name_id))
	
	label = int(label_input_file.readline())
	if len(sentence) <= SENTENCE_LENGTH:
		labels.append(np.array(label))

	if len(sentence) > 1000:
		print utterance

MAX_LABEL = 2

print('Reading ends.')
exit()

labels = to_categorical(labels, num_classes = MAX_LABEL)

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

standard_output = open("std.txt", "w")
for item in labels_valid:
	standard_output.write(str(item) + '\n')

script_input = Input(shape=(SENTENCE_LENGTH, ))
script_embedding = Embedding(VOCABULARY_SIZE, output_dim = EMBEDDING_DIM, weights = [word_vector_matrix])(script_input) 
model = Model(inputs=script_input, outputs=script_embedding)
print(model.input_shape)
print(model.output_shape)
script_output = Bidirectional(LSTM(NUM_HIDDEN), merge_mode='concat')(script_embedding)

model = Model(inputs=script_input, outputs=script_output)

print(model.input_shape)
print(model.output_shape)

name_input = Input(shape=(1,))
name_output = Embedding(NUMBER_OF_NAMES, output_dim = EMBEDDING_DIM)(name_input)
name_output = Flatten()(name_output)

model = Model(inputs=name_input, outputs=name_output)
print(model.input_shape)
print(model.output_shape)
model.summary()

utterance_input = Concatenate(axis = -1)([script_output, name_output])

#output = Dense(NUM_HIDDEN, activation ='sigmoid')(utterance_input)
output = Dense(LABEL_SIZE, activation='softmax')(utterance_input)
print output.shape

#model = Model(inputs=[script_input, name_input], outputs=output)
model = Model(inputs = [script_input, name_input], outputs = output)
print(model.input_shape)
print(model.output_shape)
model.summary()

model.compile(optimizer = SGD(lr = 0.1), loss='binary_crossentropy', metrics=['accuracy'])

print scripts_train.shape
print names_train.shape
print labels_train.shape

model.fit([scripts_train, names_train], labels_train, epochs=2, batch_size=40)
score, acc = model.evaluate([scripts_valid, names_valid], labels_valid, batch_size = 20)

print('Test score:', score)
print('Test accuracy:', acc)

OUTPUT_FILE = open("output.txt", "w")
prediction = model.predict([scripts_valid, names_valid], batch_size = 40, verbose = 0)
for item in prediction:
	print item
	if item[0] > item[1]:
		OUTPUT_FILE.write("0\n")
	else:
		OUTPUT_FILE.write("1\n")

