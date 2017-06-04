from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load train dataset
datasetTrain = numpy.loadtxt("pv_train_part1.csv", delimiter=",")
datasetTrainTest = numpy.loadtxt("pv_train_part2.csv", delimiter=",")

# load test dataset
datasetTest = numpy.loadtxt("pv_testSample.csv", delimiter=",")

# features
X = datasetTrain[:, 0:8]
# target
Y = datasetTrain[:, 8]

# features
X1 = datasetTrainTest[:, 0:8]
# target
Y1 = datasetTrainTest[:, 8]

# testfeatures
Z = datasetTest[:, 0:8]

# create model
model = Sequential()
model.add(Dense(8, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(16, init='uniform', activation='relu'))
model.add(Dense(32, init='normal', activation='relu'))
model.add(Dense(16, init='normal', activation='relu'))
model.add(Dense(8, init='normal', activation='relu'))
model.add(Dense(4, init='normal', activation='relu'))
model.add(Dense(2, init='normal', activation='relu'))
model.add(Dense(1, init='normal', activation='relu'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(X1, Y1, nb_epoch=200, batch_size=1000, validation_split=0.2, verbose = 2, shuffle = True)


# Evaluate the network
loss, accuracy = model.evaluate(X, Y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

# Make predictions
anomalies = 0

probabilities = model.predict(Z)
predictions = [float(numpy.round(x)) for x in probabilities]
print ("Predictions: ", predictions)

accuracy = numpy.mean(predictions == Y)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))

dfPrediction = pandas.DataFrame(predictions)
dfPrediction.to_csv('pv_prediction.csv', header=False, index=False)


# predictions
predictedAnomalies = predictions.count(1)


print 'Predicted Anomalies: ', predictedAnomalies


pv_predict = predictions
pv_test = pandas.read_csv('pv_test.csv', engine='python')

timestamp = pandas.DataFrame(pv_test, columns = ['timestamp'])
pageview = pandas.DataFrame(pv_test, columns = ['value'])
label = pandas.DataFrame(pv_predict, columns = ['label'])

write = pandas.concat([timestamp, pageview, label], axis=1)
write.to_csv('pv_final_result.csv', index=False)

print 'final result packed'
