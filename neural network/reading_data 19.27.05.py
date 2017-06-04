import pandas
import random


datasetTrain = pandas.read_csv('pv_train.csv', engine='python')

df0Train = pandas.DataFrame(datasetTrain, columns = ['Time series decomposition(win=5weeks)'])
df1Train = pandas.DataFrame(datasetTrain, columns = ['Historical MAD(win=2weeks)'])
df2Train = pandas.DataFrame(datasetTrain, columns = ['Time series decomposition(win=4weeks)'])
df3Train = pandas.DataFrame(datasetTrain, columns = ['Time series decomposition(win=1week)'])
df4Train = pandas.DataFrame(datasetTrain, columns = ['Diff value(to last week)'])
df5Train = pandas.DataFrame(datasetTrain, columns = ['Historical average(win=5weeks)'])
df6Train = pandas.DataFrame(datasetTrain, columns = ['Time series decomposition(win=3weeks)'])
df7Train = pandas.DataFrame(datasetTrain, columns = ['SVD(r10 c5)'])
dfLabelTrain = pandas.DataFrame(datasetTrain, columns = ['label'])

columnsTrain = pandas.concat([df0Train, df1Train, df2Train, df3Train, df4Train, df5Train, df6Train, df7Train, dfLabelTrain], axis=1)


#df = pandas.DataFrame(columnsTrain, dtype=float)
#df = df.fillna(method='ffill', inplace=False)

df1 = columnsTrain[dfLabelTrain.label == 0].sample(n=133633)
#df2 = columnsTrain[dfLabelTrain.label == 1].sample(n=6282)
df3 = columnsTrain[dfLabelTrain.label == 0].sample(n=20844)
df4 = columnsTrain[dfLabelTrain.label == 1].sample(n=12564)
#df5=pandas.concat([df1,df2],ignore_index=True)
df6=pandas.concat([df3,df4],ignore_index=True)

df1.to_csv('pv_train_part1.csv', header=False, index=False)
df6.to_csv('pv_train_part2.csv', header=False, index=False)


datasetTest = pandas.read_csv('pv_test.csv', engine='python')

df0Test = pandas.DataFrame(datasetTest, columns = ['Time series decomposition(win=5weeks)'])
df1Test = pandas.DataFrame(datasetTest, columns = ['Historical MAD(win=2weeks)'])
df2Test = pandas.DataFrame(datasetTest, columns = ['Time series decomposition(win=4weeks)'])
df3Test = pandas.DataFrame(datasetTest, columns = ['Time series decomposition(win=1week)'])
df4Test = pandas.DataFrame(datasetTest, columns = ['Diff value(to last week)'])
df5Test = pandas.DataFrame(datasetTest, columns = ['Historical average(win=5weeks)'])
df6Test = pandas.DataFrame(datasetTest, columns = ['Time series decomposition(win=3weeks)'])
df7Test = pandas.DataFrame(datasetTest, columns = ['SVD(r10 c5)'])

columnsTest = pandas.concat([df0Test, df1Test, df2Test, df3Test, df4Test, df5Test, df6Test, df7Test], axis=1)

columnsTest.fillna(method='ffill', inplace=True)

testData = columnsTest.head(86400)

testData.to_csv('pv_testSample.csv', header=False, index=False)

