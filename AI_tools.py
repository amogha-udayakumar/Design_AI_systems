#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import copy

from sklearn.model_selection import train_test_split

beijing = pd.read_csv("Cities/Beijing_labeled.csv").astype("float")
shenyang = pd.read_csv("Cities/Shenyang_labeled.csv").astype("float")

guan = pd.read_csv("Cities/Guangzhou_labeled.csv").astype("float")
shan = pd.read_csv("Cities/Shanghai_labeled.csv").astype("float")


# In[4]:


# Simple function we call to normalize the data
def normalize(df):
  for i in df.columns:
    m = df[i].mean()
    std = df[i].std()
    df[i] = (df[i]-m)/std


# In[2]:


class knn:

  def __init__(self, n):
    self.data = None
    self.labels = None
    self.n = n

  def fit(self, train, labels):
    # The data and labels are saved within the model
    self.data = train
    self.labels = labels

  def predict(self, to_predict):
    predictions = []
    # We loop through every data point to predict
    for i in to_predict:
      distances = []
      tmp_distances = []
      # We compute the distance between the given data point i and all of the known (training) data.
      for x in self.data:
        distances.append(np.linalg.norm(i-x))
        tmp_distances.append(distances[-1])

      # We sort the distances
      distances.sort()
      tmp_predictions = []

      # We get the n closest ones
      for ind, dist in enumerate(distances):
        if ind == self.n:
          break 

        # We get the index from the chosen data point in the unsorted list
        # Then we append the label of that point to the list
        # The distance of the chosen data point is set to -1 to avoid collisions
        # If we have 3 points in our data that have exactly the same distance, setting the distance to -1 ensure we do not always get the first occuring element.

        index = tmp_distances.index(dist)
        tmp_predictions.append(self.labels[index])
        tmp_distances[index] = -1    
      
      # We take the mean of the predictions and round it. So Knn, is based on majority voting. 
      m = np.mean(tmp_predictions)
      predictions.append(round(m))

    return predictions

  def score(self, test, labels):

    # We decided to use the F1 score as a metric
    pred = self.predict(test)

    cm = [[0,0], [0,0]]
    for i in range(0, len(pred)):
      cm[int(labels[i])][int(pred[i])] += 1

    print(f"Confusion matrix: {cm}\n")

    precision_0 = cm[0][0]/(cm[0][1]+cm[0][0])
    recall_0 = cm[0][0]/(cm[1][0]+cm[0][0])
    f1_0 = 2*precision_0*recall_0/(precision_0+recall_0)

    print("-- Score 0 --\n")
    print(f"F1 score: {f1_0}")
    print(f"Precision score: {precision_0}")
    print(f"Recall score: {recall_0}\n")

    precision_1 = cm[1][1]/(cm[1][0]+cm[1][1])
    recall_1 = cm[1][1]/(cm[0][1]+cm[1][1])
    f1_1 = 2*precision_1*recall_1/(precision_1+recall_1)

    print("-- Score 1 --\n")
    print(f"F1 score: {f1_1}")
    print(f"Precision score: {precision_1}")
    print(f"Recall score: {recall_1}\n")

    mean = (f1_0+f1_1)/2
    print(f"Average F1 score {mean}\n")


# In[8]:


# We balance the training data
train = beijing.append(shenyang, ignore_index=True)

train_0 = train[train["PM_HIGH"]==0].sample(n=796, random_state=42)
train_1 = train[train["PM_HIGH"]==1]

train = train_0.append(train_1, ignore_index=True)

#mean_train = train[train["PM_HIGH"]==0].mean()
#print(train[train["PM_HIGH"]==0].std())

#sns.pairplot(train, hue="PM_HIGH")

labels_train = train.loc[:, 'PM_HIGH']
del train["PM_HIGH"]

normalize(train)

# We used sklearn to split our data in train/validation sets
train_data, val_data, train_labels, val_labels = train_test_split(train, labels_train, test_size=0.3, random_state=42)

# This is the code we used to fine-tune the amount of neighbours to be taken into account
#for i in [1, 5, 10, 15, 20, 25, 30]:
#  print(i)
#  m = knn(i)
#  m.fit(train_data.to_numpy(), train_labels.to_numpy())
#  m.score(val_data.to_numpy(), val_labels.to_numpy())

#mean_guan = guan[guan["PM_HIGH"]==0].mean()
#print(guan[guan["PM_HIGH"]==0].std())

guan_labels = guan.loc[:, 'PM_HIGH']
del guan["PM_HIGH"]

normalize(guan)

#mean_shan = shan[shan["PM_HIGH"]==0].mean()
#print(shan[shan["PM_HIGH"]==0].std())

shan_labels = shan.loc[:, 'PM_HIGH']
del shan["PM_HIGH"]

normalize(shan)

model = knn(10)
model.fit(train_data.to_numpy(), train_labels.to_numpy())
print(f"- Guangzhou -\n")
model.score(guan.to_numpy(), guan_labels.to_numpy())
print(f"- Shanghai -\n")
model.score(shan.to_numpy(), shan_labels.to_numpy())


# In[ ]:




