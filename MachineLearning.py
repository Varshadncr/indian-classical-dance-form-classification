#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import os
import random
from shutil import copyfile
     


# In[2]:


train_path = "train"
test_path = "test"

labels = pd.read_csv("train .csv")
t_labels = pd.read_csv("test .csv")
     


# In[3]:


labels.head()


# In[4]:


labels.shape


# In[5]:


labels['target'].value_counts().plot(kind='bar')


# In[11]:


import os
import shutil
import pandas as pd

# Load your CSV file
train_csv_path = "train .csv"  # Replace with the actual path to your train.csv
train_path = "train"  # Replace with the actual path where your images are stored

# Load CSV into a DataFrame
labels = pd.read_csv(train_csv_path)

# Iterate through each row in the DataFrame
for idx, row in labels.iterrows():
    filename = row['Image']  # Assuming 'image' is the column with filenames
    target = row['target']   # Assuming 'target' is the column with labels
    path = os.path.join("/content/dataset/", str(target))
    if not os.path.exists(path):
        os.makedirs(path)
    shutil.copy(os.path.join(train_path, filename), path)


# In[12]:





# In[13]:


fig,ax = plt.subplots(1,5,figsize = (15,3))

# To display images when subfolders were not created
for i,idx in enumerate(labels[labels.target == 'manipuri']['Image'][-5:]):
    path = os.path.join(train_path,idx)
    ax[i].imshow(img.imread(path))

# i = 0
# fig,ax = plt.subplots(1,5,figsize = (15,3))

# for f in os.listdir(train_path + "manipuri/"):
#     # img.imread(f)
#     if i == 5:  break
#     ax[i].imshow(img.imread(train_path + "manipuri/" + f))
#     i += 1
     


# In[14]:


labels.target.unique()


# In[15]:


# # For creating validation and train folders of the images randomly for training the model

# import split_folders

# split_folders.ratio('/content/dataset/', output="/content/", seed=1337, ratio=(.8, .2))


# In[30]:


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Path to your CSV file
csv_file = "train .csv"

# Load CSV into a DataFrame
df = pd.read_csv(csv_file)

# Assuming your CSV has columns 'image' for image paths and 'target' for category labels
category_names = sorted(df['target'].unique())  # Get unique category names
img_pr_cat = df['target'].value_counts().sort_index()  # Count images per category and sort by category name

plt.figure(figsize=(10, 6))  # Adjust figure size as needed
sns.barplot(y=img_pr_cat.index, x=img_pr_cat.values)
plt.title("Number of training images per category")
plt.xlabel("Number of Images")
plt.ylabel("Categories")
plt.show()


# In[38]:


import os
import matplotlib.pyplot as plt
import matplotlib.image as img

# Define your training data directory
train_data_dir = "train"

# Iterate through directories and files in the training data directory
for subdir, dirs, files in os.walk(train_data_dir):
    for file in files:
        # Construct the full path to the image file
        img_file = os.path.join(subdir, file)
        
        # Load the image using matplotlib.image
        image = img.imread(img_file)
        
        # Display the image using matplotlib.pyplot
        plt.figure()
        plt.title(subdir)  # Set the title as the directory name
        plt.imshow(image)
        plt.axis('off')  # Turn off axis labels
        break  # Display only the first image from each directory

plt.show()


# In[48]:


# from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.applications.inception_resnet_v2   import preprocess_input, InceptionResNetV2


# In[53]:


import cv2

def load_data(df, path):
    images = []
    labels = []
    for i in zip(df.values):
        file = i[0][0]
        label = i[0][1]
        image = cv2.resize(cv2.imread(path+file), 
                           (256,256))
        image = preprocess_input(image)
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)


# In[56]:


import cv2

def load_data(df, path):
    images = []
    labels = []
    
    for i in df.iterrows():
        file = i[0][0]
        label = i[0][1]
        
        try:
            image_path = path + file
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error loading image: {image_path}")
                continue
            
            image = cv2.resize(image, (256, 256))
            image = preprocess_input(image)
            images.append(image)
            labels.append(label)
        
        except Exception as e:
            print(f"Error processing image {file}: {str(e)}")
            continue
    
    return images, labels


# In[59]:


import pandas as pd
import cv2

def load_data(df, path):
    images = []
    labels = []
    
    for index, row in df.iterrows():
        file = row['image']  # Assuming 'image' is the column containing file names
        label = row['target']  # Assuming 'target' is the column containing labels
        
        try:
            image_path = path + file
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error loading image: {image_path}")
                continue
            
            image = cv2.resize(image, (256, 256))
            image = preprocess_input(image)
            images.append(image)
            labels.append(label)
        
        except Exception as e:
            print(f"Error processing image {file}: {str(e)}")
            continue
    
    return images, labels


# In[64]:


import pandas as pd
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input

def load_data(df, path):
    images = []
    labels = []
    
    print(f"Columns in DataFrame: {df.columns}")  # Debugging print to check column names
    
    for index, row in df.iterrows():
        file = row['image']  # Assuming 'image' is the column containing file names
        label = row['target']  # Assuming 'target' is the column containing labels
        
        try:
            image_path = path + file
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error loading image: {image_path}")
                continue
            
            image = cv2.resize(image, (256, 256))
            image = preprocess_input(image)
            images.append(image)
            labels.append(label)
        
        except Exception as e:
            print(f"Error processing image {file}: {str(e)}")
            continue
    
    return images, labels


# In[65]:


num_classes = len(np.unique(y))


# In[66]:


from tensorflow.keras import layers, models, optimizers, callbacks, Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Activation,Dropout, Conv2D, GlobalAveragePooling2D
     


# In[67]:


# x1 = layers.Dense(1024, activation='relu')(model.output)
# x1 = layers.Dropout(rate=0.2)(x1)
# x2 = layers.Dense(512, activation='relu')(x1)
# x2 = layers.Dropout(rate=0.2)(x2)
# x3 = layers.Dense(128, activation='relu')(x2)

# output = layers.Dense(num_classes, activation='softmax')(x2)

# model = models.Model(model.input, output)

# model.summary()


# In[68]:


def plot_graph(hist):
  plt.figure(figsize=(15,7))
  ax1 = plt.subplot(1,2,1)
  ax1.plot(hist.history['loss'], color='b', label='Training Loss') 
  ax1.plot(hist.history['val_loss'], color='r', label = 'Validation Loss',axes=ax1)
  legend = ax1.legend(loc='best', shadow=True)
  ax2 = plt.subplot(1,2,2)
  ax2.plot(hist.history['accuracy'], color='b', label='Training Accuracy') 
  ax2.plot(hist.history['val_accuracy'], color='r', label = 'Validation Accuracy')
  legend = ax2.legend(loc='best', shadow=True)


# In[38]:


import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input

# Define your data loading function
def load_data(df, path):
    images = []
    labels = []
    
    print(f"Columns in DataFrame: {df.columns}")  # Debugging print to check column names
    
    for index, row in df.iterrows():
        file = row['Image']  # Assuming 'Image' is the column containing file names
        label = row['target']  # Assuming 'target' is the column containing labels
        
        image_path = os.path.join(path, file)  # Join path correctly
        print(f"Processing image: {image_path}")  # Print the full path for debugging
        
        if not os.path.exists(image_path):
            print(f"Image does not exist: {image_path}")
            continue
        
        try:
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error loading image: {image_path}")
                continue
            
            image = cv2.resize(image, (256, 256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = preprocess_input(image)
            images.append(image)
            labels.append(label)
        
        except Exception as e:
            print(f"Error processing image {file}: {str(e)}")
            continue
    
    return images, labels

# Assuming 'df' is your DataFrame containing image paths and labels
# Load data
path_to_images = 'train'  # Replace with your actual image directory path
df = pd.read_csv('train .csv')  # Replace with your actual DataFrame CSV file path
images, labels = load_data(df, path_to_images)

# Check if images and labels have been loaded correctly
if len(images) == 0 or len(labels) == 0:
    raise ValueError("No images or labels were loaded. Please check the file paths and ensure the images exist.")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to categorical format if not already
num_classes = len(np.unique(labels))  # Assuming num_classes is determined from your data
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# Define your model architecture
lr = 0.0001
batch_size = 64
freeze_layer = 30
epochs = 30

base_model = InceptionResNetV2(include_top=False, weights='imagenet',
                               pooling='avg', input_shape=(256, 256, 3))

for layer in base_model.layers[:freeze_layer]:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(learning_rate=lr)
model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=['accuracy'])

# Define callbacks for training
lr_reduce = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                        patience=2,
                                        verbose=1,
                                        factor=0.5,
                                        min_lr=0.00001)

model_checkpoint_callback = callbacks.ModelCheckpoint(filepath="/content/weights.h5",
                                                      save_weights_only=True,
                                                      monitor='val_loss',
                                                      mode='max',
                                                      save_best_only=True)

# Train the model
hist = model.fit(np.array(X_train), np.array(y_train),
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(np.array(X_val), np.array(y_val)),
                 callbacks=[lr_reduce, model_checkpoint_callback])

# Plot training history
def plot_graph(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.show()

plot_graph(hist)


# In[4]:


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Convert X_val to a NumPy array
X_val = np.array(X_val)

# Reshape X_val to include the channel dimension (assuming RGB images)
X_val = X_val.reshape(-1, 256, 256, 3)  # Adjust the shape according to your model input requirements

# Generate predictions
y_pred = model.predict(X_val)

# Compute confusion matrix
cm = confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1))

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[5]:


def load_testdata(df, path):
    images = []
    for i in zip(df.values):
        file = i[0][0]
        image = cv2.resize(cv2.imread(path+file), 
                           (256,256))
        image = preprocess_input(image)
        images.append(image)
    return np.array(images)


# In[23]:


import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing import image

# Example: Define t_labels and test_path
# Assuming t_labels is a DataFrame with columns 'image' and 'target'
t_labels = pd.DataFrame({
    'image': ['108.jpg', '11.jpg', '112.jpg', '145.jpg', '114.jpg'],  # Replace with your test image filenames
    'target': ['manipuri', 'kathak', 'bharatanatyam', 'odissi', 'kathakali']  # Replace with your actual targets
})

test_path = 'test'  # Replace with your actual test image path

def load_testdata(labels, img_path):
    images = []
    for idx, row in labels.iterrows():
        img_full_path = os.path.join(img_path, row['image'])
        if os.path.exists(img_full_path):
            img = image.load_img(img_full_path, target_size=(256, 256))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            images.append(img)
        else:
            print(f"File not found: {img_full_path}")
    return np.vstack(images) if images else None

# Load test data
test_x = load_testdata(t_labels, test_path)
if test_x is not None:
    print("Test data loaded successfully.")
else:
    print("No valid test images found.")


# In[28]:


import os
import matplotlib.pyplot as plt
import matplotlib.image as img

# Define the path to your training data directory
train_data_dir = "train"

# Iterate through the training data directory and plot one image from each subdirectory
for subdir, dirs, files in os.walk(train_data_dir):
    for file in files:
        img_file = os.path.join(subdir, file)
        image = img.imread(img_file)
        plt.figure()
        plt.title(subdir)
        plt.imshow(image)
        break  # Remove this break statement if you want to display all images


# In[27]:


import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd

# Example: Define t_labels and test_path
# Assuming t_labels is a DataFrame with columns 'Image' and 'target'
t_labels = pd.DataFrame({
    'Image': ['108.jpg', '11.jpg', '112.jpg', '145.jpg', '114.jpg'],  # Replace with your test image filenames
    'target': ['manipuri', 'kathak', 'bharatanatyam', 'odissi', 'kathakali']  # Replace with your actual targets
})

# Define the test image path
test_path = 'test'  # Replace with your actual test image path

# Create subplots
fig, ax = plt.subplots(1, 5, figsize=(15, 3))

# Iterate over the images and plot them
for i, idx in enumerate(t_labels['Image'][0:5]):  # Adjust range if needed
    path = os.path.join(test_path, idx)
    if os.path.exists(path):
        ax[i].imshow(img.imread(path))
        ax[i].set_title(t_labels['target'][i])
        ax[i].axis('off')  # Hide the axes
    else:
        print(f"Image {path} not found")

plt.show()


# In[ ]:




