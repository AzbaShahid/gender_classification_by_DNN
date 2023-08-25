#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import os
import pandas as pd
import numpy as np


# ### loading input data

# In[2]:


dataset=[]
folder_paths=['/Users/hp/Desktop/internship/gender/Train/female',
              '/Users/hp/Desktop/internship/gender/Train/male']
for i in folder_paths:
    folder_name=os.path.basename(i)
    for file_name in os.listdir(i):
        img_path=os.path.join(i,file_name)
        if os.path.isfile(img_path):
            img=cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                resize_img=cv.resize(img,(150,150))
                flattened_img=resize_img.flatten().tolist()
                dataset.append(flattened_img+[folder_name])



# In[3]:


df = pd.DataFrame(dataset)


# In[4]:


df


# In[4]:


df.rename(columns={df.iloc[:,-1].name:'Target'},inplace=True)


# ### randomize the data

# In[5]:


#get num of rows of dataset
num_rows=len(df)
#generate permutated indices
permuted_indices=np.random.permutation(num_rows)
#generate random data
random_df=df.iloc[permuted_indices]


# In[6]:


random_df.head()


# In[7]:


from sklearn.preprocessing import LabelEncoder


# ### normalized the data

# In[8]:


x=random_df.drop('Target',axis=1)
x=x/255


# ### encode the target

# In[9]:


encoder=LabelEncoder()
y=random_df['Target']
y_encoded=encoder.fit_transform(y)
y_series=pd.Series(y_encoded,name='target')


# In[10]:


df_encoded=pd.concat([x,y_series],axis=1)
df_encoded.head()


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.models import load_model


# ### apply dnn model

# In[37]:


sq=Sequential()
sq.add(Dense(128,activation='relu',input_shape=(150*150,)))
sq.add(Dense(64,activation='relu'))
sq.add(Dense(len(encoder.classes_),activation='softmax'))
sq.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
sq.summary()


# In[38]:


trainable_param1=(150*150)*128+128
trainable_param2=128*64+64
trainable_param3=64*2+2


# In[39]:


trainable_param3


# In[40]:


total_trainable_param = trainable_param1 + trainable_param2 + trainable_param3
total_trainable_param


# In[41]:


x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)


# In[57]:


from keras.utils import to_categorical
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Train the model on the training data
history = sq.fit(x_train, y_train_encoded, batch_size=25, epochs=20, validation_split=0.2)


# In[58]:


y_pred=sq.predict(x_test)
y_prediction=np.argmax(y_pred,axis=1)

y11=encoder.inverse_transform(y_prediction)


# ### model evaluation

# In[51]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[59]:


cm = confusion_matrix(y_test, y_prediction)
sns.heatmap(cm, annot = True )
plt.show()


# In[60]:


print (cm)


# In[61]:


accuracy = accuracy_score(y_test, y_prediction)
print("Accuracy:", accuracy)


# In[64]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


# ### deployment

# In[66]:


sq.save('ann_model.keras')


# In[90]:


sq=load_model('ann_model.keras')

user_image = cv.imread("/Users/hp/Desktop/internship/pic.jpg",cv.IMREAD_GRAYSCALE)

resized=cv.resize(user_input,(150,150))

flattened_img=resized.flatten()

normalized_user_image = flattened_img / 255.0

user_input = normalized_user_image.reshape(1, -1)

array=sq.predict(user_input)

prediction=np.argmax(array,axis=1)

predicted_class=encoder.inverse_transform(prediction)[0]

image=cv.cvtColor(resized_image,cv.COLOR_BGR2RGB)

plt.imshow(image)

plt.title(predicted_class)

