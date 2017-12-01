from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

MAX_FEATURES = 10000 #Vocabulary size
BATCH_SIZE = 32
EPOCH = 20

print('Load data...')

print('Pad sequences')
x_train = sequence.pad_sequences(x_trani, maxlen = MAXLEN)
x_test = sequence.pad_sequences(x_test, maxlen = MAXLEN)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(Bidirectional(LSTM(128), merge_mode = 'concat'))
#model.add(Dropout(0.5))
model.add(Activation('softmax'))

model.compile('adam', 'categorical_crossentropy', metrics = ['accuracy'])

print('Train...')
model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCH, validation_data = [x_test, y_test])
