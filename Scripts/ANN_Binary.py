import numpy as np
from pyrsgis import raster

Playa_Image = 'Playa_Image.tif'
Playa_Training = 'Playa_Training.tif'
NonPlaya_Training = 'NonPlaya_Training.tif'

# Reading rasters as arrays
dsImage, arrImage = raster.read(Playa_Image, bands='all')
dsPlayaTraining, arrPlayaTraining = raster.read(Playa_Training)
dsNonPlayaTraining, arrNonPlayaTraining = raster.read(NonPlaya_Training)

from pyrsgis.convert import array_to_table
arrImage = array_to_table(arrImage)
arrPlayaTraining = array_to_table(arrPlayaTraining)
arrNonPlayaTraining = array_to_table(arrNonPlayaTraining)
# arrPlayaTraining = (arrPlayaTraining==1).astype(int)
# arrNonPlayaTraining = (arrNonPlayaTraining==0).astype(int)

arrImageMin = arrImage.min(0)
arrImageMax = arrImage.max(0)
arrImageRange = arrImageMax - arrImageMin
arrImage = arrImage.astype(float)

for i in range(arrImageRange.shape[0]):
    for j in range(arrImage.shape[0]):
        arrImage[j][i] = (arrImage[j][i]-arrImageMin[i])/arrImageRange[i]

featuresPlaya = arrImage[arrPlayaTraining==1]
labelsPlaya = arrPlayaTraining[arrPlayaTraining==1]

featuresNonPlaya = arrImage[arrNonPlayaTraining==0]
labelsNonPlaya = arrNonPlayaTraining[arrNonPlayaTraining==0]

features = np.concatenate((featuresPlaya, featuresNonPlaya), axis=0)
labels = np.concatenate((labelsPlaya, labelsNonPlaya), axis=0)

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(features, labels)

# Reshaping the data
xTrain = xTrain.reshape((xTrain.shape[0], 1, xTrain.shape[1]))
xTest = xTest.reshape((xTest.shape[0], 1, xTest.shape[1]))

# Defining model parameters
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1, arrImage.shape[1])),
    keras.layers.Dense(14, activation='relu'),
    keras.layers.Dense(2, activation='softmax')])

# Defining accuracy metrics and parameters
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Running the model
model.fit(xTrain, yTrain, epochs=5)

# Predicting test data 
from sklearn.metrics import confusion_matrix, precision_score, recall_score
yTestPredicted = model.predict(xTest)
yTestPredicted = yTestPredicted[:,1]

# Calculating and displaying error metrices
yTestPredicted = (yTestPredicted>0.5).astype(int)
cMatrix = confusion_matrix(yTest, yTestPredicted)
pScore = precision_score(yTest, yTestPredicted)
rScore = recall_score(yTest, yTestPredicted)

print("Confusion matrix: for 14 nodes\n", cMatrix)
print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))

dsImagePre, arrImagePre = raster.read(Playa_Image, bands='all')

arrImagePre = array_to_table(arrImagePre)

# Normalizing the test image
arrImagePreMin = arrImagePre.min(0)
arrImagePreMax = arrImagePre.max(0)
arrImagePreRange = arrImagePreMax-arrImagePreMin
arrImagePre = arrImagePre.astype(float)

for i in range(arrImagePreRange.shape[0]):
    for j in range(arrImagePre.shape[0]):
        arrImagePre[j][i] = (arrImagePre[j][i]-arrImagePreMin[i])/arrImagePreRange[i]

arrImagePre = arrImagePre.reshape((arrImagePre.shape[0], 1, arrImagePre.shape[1]))

predicted = model.predict(arrImagePre)
predicted = predicted[:,1]

# Exporting the raster file
prediction = np.reshape(predicted, (dsImage.RasterYSize, dsImage.RasterXSize))
outFile = 'Playa_Predicted_ANN.tif'
raster.export(prediction, dsImage, filename=outFile, dtype='float')
