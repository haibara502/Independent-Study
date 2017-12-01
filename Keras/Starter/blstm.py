model = Sequential()

model.add(Embedding(max_features, 128, input_length = maxlen))
model.add(Bidirectional(LSTM(64))
model.add(Dense(1, activation = 'sigmoid'))
