#HANG SIGN RECOGNITION

#importing the libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Flatten
from keras.preprocessing.image import ImageDataGenerator


#importing datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


#data preprocessing
X_train = train_data.iloc[:,1:785].values.reshape(-1,28,28,1)
Y_train = train_data.iloc[:,0].values.reshape(27455,1)
X_test = test_data.iloc[:,1:785].values.reshape(-1,28,28,1)
Y_test = test_data.iloc[:,0].values.reshape(7172,1)

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
Y_train = label_binarizer.fit_transform(Y_train)
Y_test = label_binarizer.fit_transform(Y_test)

X_train = X_train/255
X_test = X_test/255


#data augmentation to reduce overfitting
gen = ImageDataGenerator( rotation_range = 10, zoom_range = 0.1, width_shift_range = 0.1, height_shift_range = 0.1  )
gen.fit(X_train)


#CNN model
model = Sequential()

model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())

model.add(Dense(units = 512 , activation = 'relu'))

model.add(Dense(units = 24 , activation = 'softmax'))


#model optimization
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

#fitting the model
history = model.fit(gen.flow(X_train,Y_train, batch_size = 512) ,epochs = 50 , validation_data = (X_test, Y_test))

print("Accuracy of the model is : " , model.evaluate(X_test,Y_test)[1]*100 , "%")

Y_pred = model.predict(X_test)
