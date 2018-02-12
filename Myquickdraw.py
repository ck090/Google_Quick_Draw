import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# ## Load the data from the folder of .npy files
# Load the numpy array images onto the memory
arm = np.load('data/arm.npy')
bicycle = np.load('data/bicycle.npy')
book = np.load('data/book.npy')
paperClip = np.load('data/paperclip.npy')

# print number of images in dataset and numpy array size of each image
print(arm.shape)
print(bicycle.shape)
print(book.shape)
print(paperClip.shape)

# add a column with labels, 0=cat, 1=sheep, 2=book, 3=paperclip 
arm = np.c_[arm, np.zeros(len(arm))]
bicycle = np.c_[bicycle, np.ones(len(bicycle))]
book = np.c_[book, np.full(len(book),2)]
paperClip = np.c_[paperClip, np.full(len(paperClip),3)]

#Function to plot 28x28 pixel drawings that are stored in a numpy array.
#Specify how many rows and cols of pictures to display (default 4x5).  
#If the array contains less images than subplots selected, surplus subplots remain empty.
def plot_samples(input_array, rows=4, cols=5, title=''):
    fig, ax = plt.subplots(figsize=(cols,rows))
    ax.axis('off')
    plt.title(title)

    for i in list(range(0, min(len(input_array),(rows*cols)) )):      
        a = fig.add_subplot(rows,cols,i+1)
        imgplot = plt.imshow(input_array[i,:784].reshape((28,28)), cmap='gray_r', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])

#plot arm samples
plot_samples(arm, title='Sample arm drawings\n')
plot_samples(bicycle, title = 'Sample bicycle drawings\n')
plot_samples(book, title = 'Sample book drawings\n')
plot_samples(paperClip, title = 'Sample paperClip drawings\n')
print "Done plotting"

# merge the arm, bicycle, book and paperclip arrays, and split the features (X) and labels (y). Convert to float32 to save some memory.
X = np.concatenate((arm[:5000,:-1], bicycle[:5000,:-1], book[:5000,:-1], paperClip[:5000,:-1]), axis=0).astype('float32')
y = np.concatenate((arm[:5000,-1], bicycle[:5000,-1], book[:5000,-1], paperClip[:5000,-1]), axis=0).astype('float32') # the last column

# train/test split (divide by 255 to obtain normalized values between 0 and 1)
# I will use a 50:50 split, since I want to start by training the models on 5'000 samples and thus have plenty of samples to spare for testing.
X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.5,random_state=0)


# ## CNN part
# one hot encode outputs
y_train_cnn = np_utils.to_categorical(y_train)
y_test_cnn = np_utils.to_categorical(y_test)
num_classes = y_test_cnn.shape[1]

# reshape to be [samples][pixels][width][height]
X_train_cnn = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test_cnn = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# define the CNN model
def cnn_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

np.random.seed(0)
# build the model
model_cnn = cnn_model()
# Fit the model
model_cnn.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=15, batch_size=200)
# Final evaluation of the model
scores = model_cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)

print('Final CNN accuracy: ', scores[1])
# Saving the model prediction
y_pred_cnn = model.predict_classes(X_test_cnn, verbose=0)
# Finding the accuracy score
acc_cnn = accuracy_score(y_test, y_pred_cnn)
print ('CNN accuracy: ',acc_cnn)