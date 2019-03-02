from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.python.keras.optimizers import Adam

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/MNIST/', one_hot=True, reshape=False)

model = Sequential()

model.add(InputLayer(input_shape=(28, 28, 1)))
model.add(Conv2D(kernel_size=3, strides=1, filters=256, padding='same', activation='relu', name='conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(kernel_size=3, strides=1, filters=128, padding='same', activation='relu', name='conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same', activation='relu', name='conv3'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

optimizer = Adam(lr=1e-4)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=mnist.train.images, y=mnist.train.labels, epochs=5, batch_size=128)
result = model.evaluate(x=mnist.test.images, y=mnist.test.labels)
print('\n\nAccuracy:', result[1])
