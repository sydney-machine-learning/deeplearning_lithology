import numpy as np
import random
# import sklearn
from pyrsgis import raster
from pyrsgis.ml import imageChipsFromArray

# Defining file names
featureFile = 'Sistan_ASTER.tif'
labelFile = 'Sistan_ASTER_Training.tif'

# Reading and normalizing input data
dsFeatures, arrFeatures = raster.read(featureFile, bands='all')
# arrFeatures = arrFeatures.astype(float)

for i in range(arrFeatures.shape[0]):
    bandMin = arrFeatures[i][:][:].min()
    bandMax = arrFeatures[i][:][:].max()
    bandRange = bandMax-bandMin
    for j in range(arrFeatures.shape[1]):
        for k in range(arrFeatures.shape[2]):
            arrFeatures[i][j][k] = (arrFeatures[i][j][k]-bandMin)/bandRange

# Creating chips using pyrsgis
features = imageChipsFromArray(arrFeatures, x_size=7, y_size=7)

# Reading and reshaping the label file
dsLabels, arrLabels = raster.read(labelFile)
arrLabels = arrLabels.flatten()

# Separating and balancing the classes
features = features[arrLabels!=0]
labels = arrLabels[arrLabels!=0]

# Defining the function to split features and labels
def train_test_split(features, labels, trainProp=0.75):
    dataSize = features.shape[0]
    sliceIndex = int(dataSize*trainProp)
    randIndex = np.arange(dataSize)
    random.shuffle(randIndex)
    train_x = features[[randIndex[:sliceIndex]], :, :, :][0]
    test_x = features[[randIndex[sliceIndex:]], :, :, :][0]
    train_y = labels[randIndex[:sliceIndex]]
    test_y = labels[randIndex[sliceIndex:]]
    return(train_x, train_y, test_x, test_y)

# Calling the function to split the data
train_x, train_y, test_x, test_y = train_test_split(features, labels)

# Creating the model
import tensorflow as tf
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=1, padding='valid', activation='relu', input_shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3])))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(48, kernel_size=1, padding='valid', activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

print(model.summary())

# Running the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=10)

# Predicting for test data 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
yTestPredicted = model.predict(test_x)

# Calculating and displaying error metrics
yTestPredicted = (yTestPredicted>0.5).astype(int)

cMatrix = confusion_matrix(test_y, yTestPredicted.argmax(axis=1))
pScore = precision_score(test_y, yTestPredicted.argmax(axis=1), average='micro')
rScore = recall_score(test_y, yTestPredicted.argmax(axis=1), average='micro')
fscore = f1_score(test_y, yTestPredicted.argmax(axis=1), average='micro')

print("Confusion matrix:\n", cMatrix)
print("\nP-Score: %.3f, R-Score: %.3f, F-Score: %.3f" % (pScore, rScore, fscore))

# Loading and normalizing a new multispectral image
dsPre, featuresPre = raster.read('Sistan_ASTER.tif')
featuresPre = featuresPre.astype(float)

for i in range(featuresPre.shape[0]):
    bandMinPre = featuresPre[i][:][:].min()
    bandMaxPre = featuresPre[i][:][:].max()
    bandRangePre = bandMaxPre-bandMinPre
    for j in range(featuresPre.shape[1]):
        for k in range(featuresPre.shape[2]):
            featuresPre[i][j][k] = (featuresPre[i][j][k]-bandMinPre)/bandRangePre

# Generating image chips from the array
new_features = imageChipsFromArray(featuresPre, x_size=7, y_size=7)

# Predicting new data and exporting the probability raster
newPredicted = model.predict(new_features)

prediction = np.reshape(newPredicted.argmax(axis=1), (dsPre.RasterYSize, dsPre.RasterXSize))

outFile = 'Sistan_Lithology_CNN.tif'
raster.export(prediction, dsPre, filename=outFile, dtype='float')
