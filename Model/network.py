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

WORD_VECTOR_FILE = '../Dataset/Supreme_Court/vector.txt'
WORD_EMBEDDING_DIM = 100
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
NUMBER_OF_UTTERANCE = 50
SENTENCE_LENGTH = 300
WORD_NUM = 100
WORD_EMBEDDING_SIZE = 100
NUM_HIDDEN = 50
NUMBER_OF_NAMES = 327
NAME_EMBEDDING_SIZE = 100
LABEL_SIZE = 2
UTTERANCE_LENGTH = 50

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

TEXT_INPUT_FILE = '../Dataset/Supreme_Court/supreme_input_half.txt'
LABEL_INPUT_FILE = '../Dataset/Supreme_Court/supreme_output.txt'
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
	num_utters = text_input_file.readline()
	if not num_utters:
		break
	num_utters = int(num_utters)
	print "Num_utters"
	print num_utters

	utterances = []
	name_one_script = []
	for i in range(num_utters):
		name = text_input_file.readline()
		utterance = text_input_file.readline()
		words = utterance.split()
		sentence = []
		for item in words:
			index = word_vector.get(item)
			if index is not None:
				sentence.append(np.array(index))
			else:
				sentence.append(np.zeros(100))
		while len(sentence) < SENTENCE_LENGTH:
			sentence.append(np.zeros(100))
		utterances.append(np.array(sentence))

		name = name[:-1]
		name_id = all_names.get(name)
		if name_id is not None:
			name_one_script.append(np.array(name_id))
		else:
			all_names[name] = len(all_names)
			name_id = all_names.get(name)
			name_one_script.append(np.array(name_id))
	if (num_utters > 50):
		continue
	print len(utterances)
	print len(utterances[0])
	print len(utterances[0][0])
	while len(utterances) < UTTERANCE_LENGTH:
		utterances.insert(0, np.zeros(shape = (SENTENCE_LENGTH, WORD_EMBEDDING_DIM)))
	scripts.append(np.array(utterances))
	while len(name_one_script) < UTTERANCE_LENGTH:
		name_one_script.append(np.array(MAX_NAME))
	names.append(np.array(name_one_script))

	label = int(label_input_file.readline())
	labels.append(np.array(label))

MAX_LABEL = 2

print('Reading ends.')

print("Shuffle the data")

print len(scripts)
print len(names)
print len(labels)

for item in scripts:
	for aitem in item:
		print len(aitem)
scripts = np.array(scripts)
indices = np.arange(scripts.shape[0])
np.random.shuffle(indices)

#Shuffle the scripts
scripts_temp = []
for i in range(scripts.shape[0]):
	scripts_temp.append(scripts[indices[i]])
scripts = np.array(scripts_temp)
print scripts.shape

#Shuffle the names
names_temp = []
names = np.array(names)
for i in range(names.shape[0]):
	names_temp.append(names[indices[i]])
names = np.array(names_temp)
print names.shape

labels_temp = []
labels = np.array(labels)
for i in range(labels.shape[0]):
	labels_temp.append(labels[indices[i]])
labels = np.array(labels_temp)

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
labels_valid = labels[-num_validation_samples]

script_input = Input(shape=(NUMBER_OF_UTTERANCE, SENTENCE_LENGTH, WORD_NUM))
script_output = TimeDistributed(Bidirectional(LSTM(NUM_HIDDEN), merge_mode='concat'))(script_input)

model = Model(inputs=script_input, outputs=script_output)

print(model.input_shape)
print(model.output_shape)

name_input = Input(shape=(NUMBER_OF_UTTERANCE, ))
name_output = Embedding(NUMBER_OF_NAMES, output_dim = EMBEDDING_DIM, input_length = NUMBER_OF_UTTERANCE)(name_input)

model = Model(inputs=name_input, outputs=name_output)
print(model.input_shape)
print(model.output_shape)
model.summary()

utterance_input = Concatenate(axis = -1)([script_output, name_output])

output = LSTM(NUM_HIDDEN)(utterance_input)
#print "lstm after concatenate size"
#print output.shape
output = Dense(LABEL_SIZE, activation='softmax')(output)
print output.shape

#model = Model(inputs=[script_input, name_input], outputs=output)
model = Model(inputs = [script_input, name_input], outputs = output)
print(model.input_shape)
print(model.output_shape)
model.summary()

model.compile(optimizer = SGD(lr = 0.1, decay = 0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print scripts_train.shape
print names_train.shape
print labels_train.shape

model.fit([scripts_train, names_train], labels_train, epochs=50, batch_size=1)
score, acc = model.evaluate([scripts_valid, names_valid], labels_test, batch_size = 1)

print('Test score:', score)
print('Test accuracy:', acc)
