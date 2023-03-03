#!/usr/bin/env python
# coding: utf-8

# Cloning the github repo!

# In[1]:


# Clone the repo
# !git clone https://github.com/qubvel/segmentation_models.git


# In[2]:


get_ipython().run_line_magic('cd', 'segmentation_models')


# In[3]:


# Visualise dependencies
# !cat requirements.txt


# In[4]:


# Install dependencies
# !pip install -r requirements.txt


# Building the Model

# In[5]:


# Importing libraries!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random


# In[6]:


# Function to load the dataset!!

def load_data(folder_name, img_size):
    
    # Extract the names of images!!
    data = pd.read_csv(folder_name + '.csv', header = None)
    names = data[0].to_numpy()
    
    # Number of images (image + mask)
    n = int(len(names) / 2)
    # List for dataset (X,Y)
    data = []
    
    for i in range(n):
        # Read the i'th image
        im = Image.open(folder_name + '/' + names[2 * i])
        im = im.resize((img_size, img_size))
        tpx = np.array(im)
        # Read the i'th mask layer
        im = Image.open(folder_name + '/' + names[2 * i + 1])
        im = im.resize((img_size, img_size))
        tpy = np.array(im)
        # Binary Mask layer!!
        tpy[tpy > 1] = 1
        tpy = tpy.astype(float)

        data.append((tpx,tpy))
    # For loop ends here!!
    
    return(data)


# In[7]:


# Dataset directory!
path = '/home/ubuntu/Dataset/Segmentation/'

# Folder names for each dataset!
pv1 = ['PV01_Brick', 'PV01_Concrete', 'PV01_SteelTile']
pv3 = ['PV03_Cropland', 'PV03_Grassland', 'PV03_Rooftop', 'PV03_SalineAlkali', 'PV03_Shrubwood', 'PV03_WaterSurface']
pv8 = ['PV08_Ground', 'PV08_Rooftop']

# Data for rooftop training!!
# pv3 = ['PV03_Rooftop']
# pv8 = ['PV08_Rooftop']


# In[8]:


# Loading the dataset

img_size = 224 # size of input image

# First dataset
dataset = load_data(path + pv1[0], img_size)

# All other datasets
pv_others = pv1[1:] + pv3 + pv8

for pv in pv_others:
    temp = load_data(path + pv, img_size)
    dataset = dataset + temp

# Shuffling the samples
random.shuffle(dataset)


# In[9]:


print("Total number of samples: " + str(len(dataset)))


# In[10]:


# Plot some samples!
np.random.seed(1)
idx = np.random.randint(low = 0, high = len(dataset), size = 5)

for i in idx:
    img, mask = dataset[i]
    plt.figure(figsize=(8, 5))
    plt.subplot(121)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title('Image ' + str(i))

    plt.subplot(122)
    plt.imshow(mask)
    plt.xticks([])
    plt.yticks([])
    plt.title('Mask ' + str(i))
    plt.show()


# In[11]:


# Importing segmentation models library
import tensorflow
from tensorflow import keras

import segmentation_models as sm

sm.set_framework('tf.keras')

keras.backend.set_image_data_format('channels_last')


# In[12]:


# DataLoader function to load each training batch lazily!
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


# In[13]:


# Choosing the backbone
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

BATCH_SIZE = 16
LR = 0.0001 # learning rate
EPOCHS = 10


# In[14]:


# Split the dataset 90%:5%:5% train:dev:test split
n = len(dataset)
m = int(n * 0.9)
v = m + int((n-m) / 2)

train = dataset[0:m]
val = dataset[m:v]
test = dataset[v:]

# Data loader for training
train_dataloader = Dataloder(train, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(val, batch_size=1, shuffle=False)


# In[25]:


# Define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid', encoder_freeze = True)

# Define optomizer
optim = keras.optimizers.Adam(LR)

# Define the loss (for our case it's binary!)
loss = sm.losses.bce_jaccard_loss

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# jaccard_loss = sm.losses.JaccardLoss()
# bce_loss = sm.losses.BinaryCELoss()
# loss = jaccard_loss + (1 * bce_loss) # To try different combination of loss functions!!

# Define the metrics
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# Compile keras model with defined optimozer, loss and metrics
model.compile(optim, loss, metrics)

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]


# In[26]:


# Fine tuning to keep the properly trained encoder weights (Note: set encoder_freeze = True)

# pretrain model decoder
model.fit(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=2, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)


# In[27]:


# Set all the layers trainable!
for layer in model.layers:
    layer.trainable = True

# Re-compile the model
optim = keras.optimizers.Adam(LR/10) # Reduce the learning rate if needed
model.compile(optim, loss, metrics)


# In[28]:


# Continue training
history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)


# In[29]:


# Plot training & validation iou_score values
plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')


# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Training IoU', 'Validation IoU'], loc='upper left')

plt.show()
# plt.savefig('loss.png')


# In[30]:


test_dataloader = Dataloder(test, batch_size=1, shuffle=False)

# load best weights
model.load_weights('best_model.h5') 

scores = model.evaluate_generator(test_dataloader)

print("Loss: {:.4}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.4}".format(metric.__name__, value))


# In[48]:


# Predictions on Test dataset!
samples = np.random.randint(low = 0, high = len(test), size = 5)

for j in samples:
    x, y = test[j]
    plt.figure(figsize=(12, 5))
    plt.subplot(131)
    plt.imshow(x)
    plt.xticks([])
    plt.yticks([])
    plt.title('Image')
    
    plt.subplot(132)
    plt.imshow(y)
    plt.xticks([])
    plt.yticks([])
    plt.title('True Mask')
    
    mask = model.predict(x.reshape(1,img_size,img_size,3)).round()
    plt.subplot(133)
    plt.imshow(mask.reshape(img_size,img_size))
    plt.xticks([])
    plt.yticks([])
    plt.title('Predicted Mask')
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




