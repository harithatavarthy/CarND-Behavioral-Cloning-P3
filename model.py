
# coding: utf-8

# In[1]:


import os
import csv

samples = []
added = 0
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        added = added + 1
        if((float(line[3])>0.5) or (float(line[3])<-0.5)):
            for repeat in range(3):
                samples.append(line)
                added = added + 1
        

print("loaded lines..",len(samples))



# In[2]:


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print("Training Sample size....",len(train_samples))
print("Validation Sample size....",len(validation_samples))


# In[3]:


import cv2
import numpy as np
import sklearn
import random

def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            correction = 0.2
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('\\')[-1]
                #print(name)
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.0)
                
                name = './data/IMG/'+batch_sample[1].split('\\')[-1]
                left_image = cv2.imread(name)
                left_angle = float(batch_sample[3])+correction
                images.append(left_image)
                angles.append(left_angle)
                images.append(cv2.flip(left_image,1))
                angles.append(left_angle*-1.0)
                
                name = './data/IMG/'+batch_sample[2].split('\\')[-1]
                right_image = cv2.imread(name)
                right_angle = float(batch_sample[3])-correction
                images.append(right_image)
                angles.append(right_angle)
                images.append(cv2.flip(right_image,1))
                angles.append(right_angle*-1.0)

                
            # trim image to only see section with road
            
            X_train = np.array(images)
            #X_train = np.reshape(X_train, X_train.shape + (160,320,3))
            y_train = np.array(angles)
            #print(X_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[4]:


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)


# In[5]:


from keras.models import Sequential,Model
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers.advanced_activations import ELU
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda((lambda x:(x/255.0) - 0.5),input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),border_mode='same',activation='elu'))
model.add(MaxPooling2D(border_mode='same'))
model.add(Convolution2D(36,5,5,subsample=(2,2),border_mode='same',activation='elu'))
model.add(MaxPooling2D(border_mode='same'))
model.add(Convolution2D(48,5,5,subsample=(2,2),border_mode='same',activation='elu'))
model.add(MaxPooling2D(border_mode='same'))
model.add(Convolution2D(64,3,3,border_mode='same',activation='elu'))
model.add(MaxPooling2D(border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,border_mode='same',activation='elu'))
model.add(MaxPooling2D(border_mode='same'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(10))
model.add(ELU())
model.add(Dense(1))
#model.add(LINEAR())
#model.add(activation('softmax'))


model.compile(loss='mse',optimizer='adam')
print(model.summary())

history_object=model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3, validation_data=validation_generator,nb_val_samples=len(validation_samples)*3, nb_epoch=10)
model.save('model.h9')


# In[6]:


import matplotlib.pyplot as plt
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

