print('Build model...')

model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape = (MAX_LEN, len(chars))))
