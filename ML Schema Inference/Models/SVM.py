#Copyright 2019 Vraj Shah, Arun Kumar
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import svm
import numpy as np # linear algebra
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model


# In[2]:


#read csv
dict_label = {'Usable directly numeric':0, 'Usable with extraction':1, 'Usable with Extration': 1, 'Usable with extraction ':1, 'Usable directly categorical':2, 'Unusable':3, 'Context_specific':4, 'Usable directly categorical ':2}
data = pd.read_csv('data_for_ML_num.csv')

data['y_act'] = [dict_label[i] for i in data['y_act']]
y = data.loc[:,['y_act']]

data = data.rename(columns={'Num of nans': 'Num_of_nans', 'num of dist_val': 'num_of_dist_val'})

data['Num_of_nans'] = [float(data['Num_of_nans'][i])/float(data['Total_val'][i]) for i in data.index]
data['num_of_dist_val'] = [float(data['num_of_dist_val'][i])/float(data['Total_val'][i]) for i in data.index]

data1 = data[['Num_of_nans', 'max_val', 'mean', 'min_val', 'num_of_dist_val','std_dev','castability','extractability', 'len_val']]
data1 = data1.fillna(0)

data1 = data1.rename(columns={'mean': 'scaled_mean', 'min_val': 'scaled_min_val', 'max_val': 'scaled_max_val','std_dev': 'scaled_std_dev'})

data1.to_csv('before.csv')

data1.loc[data1['scaled_min_val'] > 10000, 'scaled_min_val'] = 10000
data1.loc[data1['scaled_min_val'] < -10000, 'scaled_min_val'] = -10000
data1.loc[data1['scaled_max_val'] > 10000, 'scaled_max_val'] = 10000
data1.loc[data1['scaled_max_val'] < -10000, 'scaled_max_val'] = -10000
data1.loc[data1['scaled_mean'] > 1000, 'scaled_mean'] = 1000
data1.loc[data1['scaled_mean'] < -1000, 'scaled_mean'] = -1000
data1.loc[data1['scaled_std_dev'] > 1000, 'scaled_std_dev'] = 1000
data1.loc[data1['scaled_std_dev'] < -1000, 'scaled_std_dev'] = -1000
        
data1.to_csv('after.csv')

column_names_to_normalize = ['scaled_max_val', 'scaled_mean', 'scaled_min_val','scaled_std_dev']
# column_names_to_normalize = ['scaled_mean','scaled_std_dev', 'scaled_len_val']
x = data1[column_names_to_normalize].values
x = np.nan_to_num(x)
x_scaled = StandardScaler().fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = data1.index)
data1[column_names_to_normalize] = df_temp

data1.Num_of_nans = data1.Num_of_nans.astype(float)
data1.num_of_dist_val = data1.num_of_dist_val.astype(float)
data1.castability = data1.castability.astype(float)
data1.extractability = data1.extractability.astype(float)
y.y_act = y.y_act.astype(float)



# In[3]:


import enchant
data1.to_csv('before.csv')
f = open('current.txt','w')
d = enchant.Dict("en_US")

for i in data.index:
    ival = data.at[i,'Attribute_name']
    if ival != 'id' and d.check(ival):
        print >> f,ival
        print >> f,y.at[i,'y_act']
        data1.at[i,'dictionary_item'] = 1
    else:
        data1.at[i,'dictionary_item'] = 0

data1.to_csv('after.csv')
f.close()
from sklearn.feature_extraction.text import CountVectorizer
# print(data1.columns)

arr = data['Attribute_name'].values
data = data.fillna(0)
arr1 = data['sample_1'].values
arr1 = [str(x) for x in arr1]
arr2 = data['sample_2'].values
arr2 = [str(x) for x in arr2]

# print(arr)
# print(arr1)
vectorizer = CountVectorizer(ngram_range=(3,3),analyzer='char')
X = vectorizer.fit_transform(arr)
X1 = vectorizer.fit_transform(arr1)
X2 = vectorizer.fit_transform(arr2)

print(len(vectorizer.get_feature_names()))

data1.to_csv('before.csv')
tempdf = pd.DataFrame(X.toarray())
tempdf1 = pd.DataFrame(X1.toarray())
tempdf2 = pd.DataFrame(X2.toarray())

data2 = pd.concat([data1,tempdf,tempdf1,tempdf2], axis=1, sort=False)

# data2.to_csv('after.csv')


# In[4]:


X_train, X_test,y_train,y_test = train_test_split(data2,y, test_size=0.2,random_state=100)


print(data1.mean())
print(data1.median())
print(data1.std())


X_train_new = X_train.reset_index(drop=True)
y_train_new = y_train.reset_index(drop=True)

X_train_new = X_train_new.values
y_train_new = y_train_new.values

k = 5
kf = KFold(n_splits=k)
avg_train_acc,avg_test_acc = 0,0
    
cvals = [0.1,1,10,100,1000]
gamavals = [0.0001,0.001,0.01,0.1,1,10]

# cvals = [1]
# gamavals = [0.0001]

bestPerformingModel = svm.SVC(C=100,decision_function_shape="ovo", gamma = 0.001)
bestscore = 0
# for cval in cvals:
#     for gval in gamavals:
#         clf = svm.SVC(C = cval,decision_function_shape="ovo", gamma = gval)
#         avgsc = 0
#         for train_index, test_index in kf.split(X_train_new):
#             X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
#             y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]

#             clf.fit(X_train_cur, y_train_cur)
#             sc = clf.score(X_test_cur, y_test_cur)
#             avgsc = avgsc + sc
#         avgsc = avgsc/k
#         print(avgsc)
#         if bestscore < avgsc:
#             bestscore = avgsc
#             bestPerformingModel = clf
#             print(bestPerformingModel)


avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
avgsc,avgsc_train,avgsc_hld = 0,0,0

for train_index, test_index in kf.split(X_train_new):
    X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
    y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]
    X_train_train, X_val,y_train_train,y_val = train_test_split(X_train_cur,y_train_cur, test_size=0.25,random_state=100)

    bestPerformingModel = svm.SVC(C=100,decision_function_shape="ovo", gamma = 0.001)
    bestscore = 0
    for cval in cvals:
        for gval in gamavals:
            clf = svm.SVC(C = cval,decision_function_shape="ovo", gamma = gval)
            clf.fit(X_train_train, y_train_train)
            sc = clf.score(X_val, y_val)

            if bestscore < sc:
                bestscore = sc
                bestPerformingModel = clf
#                 print(bestPerformingModel)

    bscr_train = bestPerformingModel.score(X_train_cur, y_train_cur)
    bscr = bestPerformingModel.score(X_test_cur, y_test_cur)
    bscr_hld = bestPerformingModel.score(X_test, y_test)

    avgsc_train_lst.append(bscr_train)
    avgsc_lst.append(bscr)
    avgsc_hld_lst.append(bscr_hld)
    
    avgsc_train = avgsc_train + bscr_train    
    avgsc = avgsc + bscr
    avgsc_hld = avgsc_hld + bscr_hld

    print(bscr_train)
    print(bscr)
    print(bscr_hld)


# In[5]:


print(avgsc_train_lst)
print(avgsc_lst)
print(avgsc_hld_lst)

print(avgsc_train/k)
print(avgsc/k)
print(avgsc_hld/k)

y_pred = bestPerformingModel.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print('Confusion Matrix: Actual (Row) vs Predicted (Column)')
print(cnf_matrix)


# In[8]:


print("Class 0: Usable directly numeric")
print("Class 1: Usable with Extraction")
print("Class 2: Usable directly categorical")
print("Class 3: Unusable")
print("Class 4: Context_specific")


# In[ ]:




