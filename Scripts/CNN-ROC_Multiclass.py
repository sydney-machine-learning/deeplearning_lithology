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
arrFeatures = arrFeatures.astype(float)

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
arrLabels[arrLabels==255] = 0
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
model.fit(train_x, train_y, epochs=2)

# Predicting for test data 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
yTestPredicted = model.predict(test_x)
y_score = yTestPredicted[:, 1:yTestPredicted.shape[1]]

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

# ROC curve
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Computing ROC parameters for each class
n_classes = 9
fpr = dict()
tpr = dict()
roc_auc = dict()

from sklearn.preprocessing import label_binarize
test_y = label_binarize(test_y, classes=list(range(1, n_classes+1)))

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Computing micro-average ROC parameters
fpr['micro'], tpr['micro'], _ = roc_curve(test_y.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Computing macro-average ROC parameters
# First aggregating all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolating all ROC curves at the points obtained from the line above
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally averaging and computing AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plotting
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
