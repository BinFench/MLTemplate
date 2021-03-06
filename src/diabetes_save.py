import numpy
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os

numpy.random.seed(7)

dataset = loadtxt('../data/pima-indians-diabetes.data.txt', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))

predictions = model.predict_classes(x)
for i in range(5):
	print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))

model_json = model.to_json()
with open("../model/model.json", "w") as json_file:
    json_file.write(model_json)
	
model.save_weights("../model/model.h5")
print("Saved model to disk")