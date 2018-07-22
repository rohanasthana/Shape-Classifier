import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from keras.models import Model
from keras.layers import Input
from keras.preprocessing import image
from keras.layers.convolutional import Conv1D
from keras.layers import Flatten, Reshape
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from keras.layers.merge import concatenate
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
shapes=['circles','triangles','squares']
filepath='../shape recong/shapes/'
imgdata=[]
shapedata=[]
for x in range(0,3,1):
    for i in range(1,101,1):
        file_name=filepath+shapes[x]+"/"+'drawing(' +str(i)+ ').png'
        img=image.load_img(file_name,grayscale=True)
        print(file_name)
        x1=image.img_to_array(img)
        x1=np.reshape(x1,(28,28))
        imgdata.append(x1)
        shapedata.append(x)
x=np.array(imgdata)
shapes=keras.utils.to_categorical(shapedata,3)
print(shapes.shape)
X_train,X_test,y_train,y_test=train_test_split(x,shapes,test_size=1/6,stratify=shapes)
print(X_train.shape)
print(y_train.shape)
activation_func = Activation('relu')
kernel_size=3
model=Sequential()
inputs=Input((28,28))
    # Convolutional block_1
conv1 = Conv1D(32, kernel_size)(inputs)
act1 = activation_func(conv1)
bn1 = BatchNormalization()(act1)
pool1 = MaxPooling1D(pool_size=2, strides=2)(bn1)

    # Convolutional block_2
conv2 = Conv1D(64, kernel_size)(pool1)
act2 = activation_func(conv2)
bn2 = BatchNormalization()(act2)
pool2 = MaxPooling1D(pool_size=2, strides=2)(bn2)

    # Convolutional block_3
conv3 = Conv1D(128, kernel_size)(pool2)
act3 = activation_func(conv3)
bn3 = BatchNormalization()(act3)
    
    # Global Layers
gmaxpl = GlobalMaxPooling1D()(bn3)
gmeanpl = GlobalAveragePooling1D()(bn3)
mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)

    # Regular MLP
dense1 = Dense(512,
kernel_initializer='glorot_normal',
bias_initializer='glorot_normal')(mergedlayer)
actmlp = activation_func(dense1)
reg = Dropout(0.5)(actmlp)

dense2 = Dense(512,
        kernel_initializer='glorot_normal',
        bias_initializer='glorot_normal')(reg)
actmlp = activation_func(dense2)
reg = Dropout(0.5)(actmlp)
    
dense2 = Dense(3, activation='softmax')(reg)
model=Model(input=inputs,output=[dense2])


sgd = keras.optimizers.SGD(lr=0.001, momentum=0.99, decay=1e-5, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
      optimizer=sgd,
      metrics=['accuracy'])
history = model.fit(X_train, y_train,
      batch_size=128,
      epochs=1000,
      verbose=1)
score=model.evaluate(X_test,y_test,verbose=0)
print('Test accuracy:',score[1])
plt.plot(history.history['acc'])
plt.show()
plt.plot(history.history['loss'])
plt.show()






