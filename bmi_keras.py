
# coding: utf-8

# In[4]:

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np


# In[5]:

csv = pd.read_csv('bmi.csv')
csv['height'] /= 200
csv['weight'] /= 100
csv


# In[7]:

# pandasのままでは使えないので、ndarray形式に変換
X = csv[['height', 'weight']].as_matrix()

# In[8]:

bmi_dic = {'thin':[1,0,0], 'normal':[0,1,0], 'fat':[0,0,1]}


# In[9]:

y = np.empty((20000,3))
for i, v in enumerate(csv['label']):
    y[i] = bmi_dic[v]


# In[16]:

# データを訓練用とテスト用に分ける
X_train, y_train = X[1:15001],  y[1:15001]
X_test, y_test = X[15001:20001],  y[15001:20001]


# In[ ]:

# モデルの構造を定義
model = Sequential()
model.add(Dense(512, input_shape=(2, )))
model.add(Activation('relu'))
model.add(Dropout(0.1))


# In[ ]:

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))


# In[ ]:

model.add(Dense(3))
model.add(Activation('softmax'))


# In[ ]:

# モデルを構築
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'rmsprop',
    metrics = ['accuracy']
)


# In[ ]:

# 訓練
hist = model.fit(
    X_train,
    y_train,
    batch_size = 100,
    nb_epoch = 20,
    validation_split = 0.1,
    callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)],
    verbose = 1
)


# In[ ]:

# テストして評価
score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])
