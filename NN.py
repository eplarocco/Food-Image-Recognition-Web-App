# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 22:39:39 2022

@author: Eileanor
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import Input
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
base_dir = "./output"
train_dataset = image_dataset_from_directory(
    os.path.join(base_dir, 'train'),
    image_size=(180, 180),
    batch_size=32)
validation_dataset = image_dataset_from_directory(
    os.path.join(base_dir, 'val'),
    image_size=(180, 180),
    batch_size=32)
test_dataset = image_dataset_from_directory(
    os.path.join(base_dir, 'test'),
    image_size=(180, 180),
    batch_size=32)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1), #0.1 is the limit, picks random num btwn 0 and 0.1
        layers.RandomZoom(0.2),
    ]
)

#Plot sample of augmented data
"""
plt.figure(figsize=(10, 10))
for images, _ in train_dataset.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
"""
       
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Input
model = models.Sequential()
model.add(Input(shape=(180, 180, 3))) #standardize input size, RGB
model.add(data_augmentation)
model.add(layers.Rescaling(1./255)) #scale images
#Feature Detector
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
#Classifier
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(6, activation='softmax')) #6 classes/ 6 food types - need to dynamically change with number of folders
#Summary
model.summary()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Model Configurations
model.compile(loss="sparse_categorical_crossentropy",
optimizer=optimizers.RMSprop(learning_rate=1e-4),
metrics=["accuracy"])

#History
history = model.fit(
      train_dataset,
      epochs=50,
      validation_data=validation_dataset)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Save Model
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# serialize model to JSON (architecture/configuration)
model_json = model.to_json()
with open("model.json", "w") as json_file :
	json_file.write(model_json)

# serialize weights to HDF5 (weights created during training)
model.save_weights("model.h5")
print("Saved model to disk")

#save full model to load later for use in predicting
model.save('CNN.model')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Diagnostic Plots
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training/Validation Accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()

#Loss
plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training/Validation loss')
plt.ylabel('Loss value')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()
