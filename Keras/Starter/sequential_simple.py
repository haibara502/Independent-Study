from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([Dense(32, units=784), Activation('relu'), Dense(10), Activation('softmax'),])
#model = Sequential()
#model.add(Dense(32, input_shape = (784,)))
#model.add(Activation('relu'))


#model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
#model.compile(optimizer = 'rmsprop', los = 'mse')

import keras.backend as K

def mean_pred(y_true, y_pred):
	return K.mean(y_pred)

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy', mean_pred])

model.fit(data, labels, epochs = 10, batch_size = 32)
