from keras.datasets import mnist
print("h")
dataset = mnist.load_data('mymnist.db')

len(dataset)

dataset

train , test = dataset

type(train)

X_train , y_train = train

X_train.shape

X_test , y_test = test

X_test.shape

img1 = X_train[7]

img1.shape

import cv2

img1_label = y_train[7]

img1_label

img1.shape

#import matplotlib.pyplot as plt

#plt.imshow(img1 , cmap='gray')

#img1.shape

img1_1d = img1.reshape(28*28)

img1_1d.shape

X_train.shape

X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)

X_train_1d.shape

X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')

X_train.shape


X_train = X_train.reshape(-1,28,28,1)

from keras.utils.np_utils import to_categorical

y_train_cat = to_categorical(y_train)

y_train_cat

#y_train_cat[7]

from keras.models import Sequential

from keras.layers import Dense

from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = (28,28,1),activation = 'relu'))

model.summary()

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=256, activation='relu'))

model.summary()

model.add(Dense(units=10, activation='softmax'))

model.summary()

from keras.optimizers import RMSprop

model.compile(optimizer='adam', loss='categorical_crossentropy', 
             metrics=['accuracy']
             )

model.layers.pop()

model.summary()

h = model.fit(X_train, y_train_cat,batch_size=512)
model.save("digit.h5")
