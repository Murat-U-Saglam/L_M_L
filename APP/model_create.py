import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from PIL import Image

from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  Flatten




(X_train, y_train), (X_val, Y_val) = cifar10.load_data()

#Preprocessing
X_train = X_train / 255
X_val = X_val / 255


y_train = to_categorical(y_train, 10)
Y_val = to_categorical(Y_val, 10)


model = Sequential([
    Flatten(input_shape=(32,32,3)),
    Dense(1000, activation='relu'),
    Dense(10, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, Y_val))
model.save('APP/model.keras')
