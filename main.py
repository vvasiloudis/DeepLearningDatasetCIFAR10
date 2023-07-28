from tensorflow import keras
from tensorflow.keras import models, layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical, plot_model
from keras.layers import Dropout

# Loading the dataset
CIFAR_10 = keras.datasets.cifar10

(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = CIFAR_10.load_data()

print(X_TRAIN.shape)
print(Y_TRAIN.shape)
print(X_TEST.shape)
print(Y_TEST.shape)
print(Y_TRAIN)

# Defining array. Each item of array represent integer value of labels. 10
# clothing item for 10 integer label.

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog',
               'Ship', 'Horse', 'Truck']
print(class_names)

# Display the first 25 images from training set

plt.figure(figsize=(10, 10))
for i in range(25):  # 25 images
    plt.subplot(5, 5, i+1)  # matrix of 5 x 5 array
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_TRAIN[i], cmap=plt.cm.binary)  # printing binary/black and white image
    # The CIFAR labels happen to be arrays, which is why you need the extra index
    plt.xlabel("%s %s" % (Y_TRAIN[i], class_names[Y_TRAIN[i][0]]))  # assigning name to each image
plt.show()

# Pixel value of the image falls between 0 to 255
new_x_train = X_TRAIN/255  # So, we are scale the value between 0 to 1 before by deviding each value by 255
print(new_x_train.shape)

new_x_test = X_TEST/255  # So, we are scale the value between 0 to 1 before by deviding each value by 255
print(new_x_test.shape)

# One hot encoding of the labels.
# (generally we do one hot encoding of the feature in EFA but in this case
# we are doing it for labels)

# Before one hot encoding
print("Y_TRAIN Shape: %s and value: %s" % (Y_TRAIN.shape, Y_TRAIN))
print("Y_TEST Shape: %s and value: %s" % (Y_TEST.shape, Y_TEST))

y_train = to_categorical(Y_TRAIN)
y_test = to_categorical(Y_TEST)

# After one hot encoding
print("y_train Shape: %s and value: %s" % (y_train.shape, y_train[0]))
print("y_test Shape: %s and value: %s" % (y_test.shape, y_test[1]))

# create a sequential model i.e. empty neural network which has no layers in it.
model = models.Sequential()

# ==================== Feature Detection / extraction Block ====================#

# in the first block we need to mention input_shape
model.add(layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3), activation='relu'))
# Add the max pooling layer
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Add Second convolutional block
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# Add the max pooling layer
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Add Third convolutional block
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# Add the max pooling layer
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# ==================== Transition Block (from feature detection to classification) ====================#

# Add Flatten layer. Flatten simply converts matrics to array
model.add(layers.Flatten(input_shape=(32, 32)))  # this will flatten the image and after this Classification happens

# ==================== Classification Block ====================#

# Classification segment - fully connected network
# The Dence layer does classification and is deep neural network. Dense layer always accept the array.
model.add(layers.Dense(128, activation='relu'))  # as C5 layer in above image.
model.add(layers.Dense(100, activation='relu'))  # as C5 layer in above image.
model.add(layers.Dense(80, activation='relu'))  # as C5 layer in above image.

# Add the output layer
model.add(layers.Dense(10, activation='softmax'))  # as Output layer in above image.The output layer normally have softmax activation

# Compile the model

# if we use softmax activation in output layer then best fit optimizer is categorical_crossentropy
# for sigmoid activation in output layer then loss will be binary_crossentropy

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# if we do not go for One Hot Encoding then use loss='sparse_categorical_crossentropy'

model.summary()

# Train the model
# Using GPU really speeds up this code
new_x_train_2 = new_x_train.reshape(50000, 32, 32, 3)
new_x_test_2 = new_x_test.reshape(10000, 32, 32, 3)

print(new_x_train.shape)
print(new_x_test.shape)
print(y_train.shape)
print(y_test.shape)

model.fit(new_x_train_2, y_train, epochs=40, batch_size=56, verbose=True, validation_data=(new_x_test_2, y_test))

# evaluate accuracy of the model
TestLoss, TestAcc = model.evaluate(new_x_test_2, y_test)
print("accuracy:", TestAcc)

# predicting lable for test_images

predictions = model.predict(new_x_test_2)

# Prediction of the 1st result. It will show the 10 predictions of labels for test image
print("1. Prediction array: %s" % (predictions[0]))

# we will verify that which result for label has high confidence
print("2. Label number having highest confidence in prediction array: %s" % (np.argmax(predictions[0])))

# let us verify what is the label in test_labels.
print("3. Actual label in dataset: %s" % (y_test[0]))

# creating a function which will help to verify the prediction is true of not
def plot_image(i, PredictionsArray, TrueLabel, img):
    # taking index and 3 arrays viz. prediction array, true label array and image array
    PredictionsArray, TrueLabel, img = PredictionsArray[i], TrueLabel[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)  # showing b/w image
    PredictedLabel = np.argmax(PredictionsArray)
    TrueLabel = np.argmax(TrueLabel)

    if PredictedLabel == TrueLabel:  # setting up label color
        color = 'blue'  # correct then blue colour
    else:
        color = 'red'  # wrong the red colour

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[PredictedLabel], 100*np.argmax(PredictionsArray), class_names[TrueLabel]), color=color)

    # function to display bar chart showing whether image prediction is how much correct
def plot_value_array(i, PredictionsArray, TrueLabel):
    PredictionsArray, TrueLabel = PredictionsArray[i], TrueLabel[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), PredictionsArray, color='gray')
    plt.ylim([0, 1])
    PredictionsLabel = np.argmax(PredictionsArray)
    TrueLabel = np.argmax(TrueLabel)

    thisplot[PredictionsLabel].set_color('red')
    thisplot[TrueLabel].set_color('green')


# verification of several images

NumRows = 6
NumCols = 5
NumImages = NumRows * NumCols
FigureCalculateNumCols = 2 * 2 * NumCols
FigureCalculateNumRows = 2 * NumRows
ImageCalculateNumCols = 2 * NumCols

plt.figure(figsize=(FigureCalculateNumCols, FigureCalculateNumRows))
for i in range(NumImages):
    plt.subplot(NumRows, ImageCalculateNumCols, 2*i+1)
    plot_image(i, predictions, y_test, new_x_test)
    plt.subplot(NumRows, ImageCalculateNumCols, 2*i+2)
    plot_value_array(i, predictions, y_test)
plt.show()

