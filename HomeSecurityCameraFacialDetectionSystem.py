#Course: Artificial intelligence
#Project: Home security camera Facial detection system 
#Authors: Jean Rene and Abdou Aziz Gajigo

#import all the needed libraries and frameworks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os 
from tensorflow.keras.optimizers import SGD

#load image here
img = load_img("/Users/abdouaziz/Desktop/Artificial Intelligence/faces/kai.jpg")

# %%
#show image using matplotlib
plt.imshow(img)

#to check the shape of this matrix we call the .shape (here we see that it has 496 height and 606 width) and it is a 3D
cv2.imread("/Users/abdouaziz/Desktop/Artificial Intelligence/faces/kai.jpg").shape


#lets generate our training and validation dataset using the imageDataGenerator
training = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale=1/255)

#lets convert our images into a dataset that can be fitted to our neuronetwork and do the same for our validation dataset
training_dataset = training.flow_from_directory("/Users/abdouaziz/Desktop/Artificial Intelligence/faces/training" , target_size= (200,200), batch_size= 3,
                                                class_mode = 'binary')

#do the same for the validation dataset
validation_dataset = validation.flow_from_directory("/Users/abdouaziz/Desktop/Artificial Intelligence/faces/validation" , target_size= (200,200), batch_size= 3,
                                                class_mode = 'binary')

#to check how they are labeled with data generatedd with the previous function we use the .class_indices function
training_dataset.class_indices

# %%
#we can see that all the baby images are labeled with 0s, all the boys images are labeled with 1s and so on
training_dataset.classes


#defining our model
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation= 'relu', input_shape = (200,200,3)), 
                                   tf.keras.layers.MaxPool2D(2,2),
                                   #increase the number of filters to 32
                                   tf.keras.layers.Conv2D(32,(3,3),activation= 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   #increase filters to 64
                                   tf.keras.layers.Conv2D(64,(3,3),activation= 'relu'), 
                                   tf.keras.layers.MaxPool2D(2,2),
                                   ##
                                   tf.keras.layers.Flatten(),
                                   ##
                                   tf.keras.layers.Dense(512,activation= 'relu'),
                                   ## set activation to sigmoid because we are using binary
                                   tf.keras.layers.Dense(1,activation= 'sigmoid')

])
model.summary() #check the summary of the model


#Now, we will train the model using the training data. We can specify the number of epochs (iterations over the dataset) and the batch size.
# Compile the model with sparse_categorical_crossentropy for multi-class classification
model.compile(loss='binary_crossentropy',
              optimizer = RMSprop(learning_rate=0.001),  # Use this for integer labels
              metrics=['accuracy'])

model_fit = model.fit(training_dataset,
                      steps_per_epoch = 3,
                      epochs = 15,
                      validation_data = validation_dataset)

# Define the directory containing the images
image_directory = '/Users/abdouaziz/Desktop/Artificial Intelligence/faces/testing'

# Get all image filenames in the directory
image_filenames = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Initialize a list to store images and predictions
images = []
predicted_labels = []

# Loop through each image file in the directory
for img_name in image_filenames:
    # Load and preprocess the image
    img_path = os.path.join(image_directory, img_name)
    img = load_img(img_path, target_size=(200, 200))  # Resize the image to (200, 200)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values to [0, 1]
    
    # Make prediction for the image
    predictions = model.predict(img_array)
    
    # Map the predicted class (0 or 1) to "Female" or "Male"
    predicted_class = int(predictions[0] > 0.5)
    class_labels = {0: "Homeowner", 1: "Stranger"}
    predicted_label = class_labels[predicted_class]
    
    # Append the original image and the predicted label to lists
    images.append(load_img(img_path, target_size=(200, 200)))  # Load image for display
    predicted_labels.append(predicted_label)

# Display the images with their predicted labels
plt.figure(figsize=(10, 10))
for i in range(len(images)):
    plt.subplot(3, 3, i + 1)  # Adjust grid size (3, 3 for up to 9 images)
    plt.imshow(images[i])
    plt.title(f"Predicted: {predicted_labels[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()


import shutil

temp_path = '/tmp/my_model.h5'
model.save(temp_path)

# Move it to your desired location
shutil.move(temp_path, '/Users/abdouaziz/Desktop/my_model.h5')
