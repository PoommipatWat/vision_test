#For Google Colab Version
#https://colab.research.google.com/drive/1ii6tcCsUQiNn19Id5FfJqGE49tSMszRO?usp=share_link

from keras.models import Sequential
from keras.layers import Input, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

#Create model by using sequential structure
model = Sequential()
model.add(Dense(5, input_dim=5, activation='tanh'))
model.add(Dense(5, activation='tanh'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#Read data from file (download at https://github.com/neokarn/computer_vision/blob/master/data.csv)
data = np.asarray([[float(num) for num in line.split(',')] for line in open('data.csv')])

#Train Model
x_train = data[0:100,0:5]
y_train = data[0:100,5]
y_train = to_categorical(y_train)

x_val = data[100:120,0:5]
y_val = data[100:120,5]
y_val = to_categorical(y_val)

h = model.fit(x_train, y_train,
          epochs=200, batch_size=5,
          validation_data=(x_val,y_val))

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['train', 'val'])

#save model 
model.save('my_model.h5')


plt.show()
