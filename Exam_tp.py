
# coding: utf-8

# In[1]:


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob


# In[2]:


import pandas as pd
import numpy as np

train = pd.read_csv('ISIC-2017_Training_Classes.csv', names=['image_id', 'melanoma'], skiprows=1)
valide = pd.read_csv('ISIC-2017_Validation_Classes.csv', names=['image_id', 'melanoma'], skiprows=1)


# In[3]:


train_im = []

path = 'ISIC-2017_Training_Data'
for f in train.image_id:
    ab_path = path+'/'+str(f)+'.jpg'
    im = mpimg.imread(ab_path)
    train_im.append(im)


# In[4]:


valide_im = []

path = 'ISIC-2017_Validation_Data'
for f in valide.image_id:
    ab_path = path+'/'+str(f)+'.jpg'
    im = mpimg.imread(ab_path)
    valide_im.append(im)


# In[5]:


from d_feautre import *

def creatFeature(images):
    res = []
    for im in images:
        fd_hu = d_hu_moments(im)
        fd_ha = d_haralick(im)
        fd_hi = d_histogram(im)
        dc = d_color_moment(im)
        res.append(np.hstack((fd_hu,fd_ha,fd_hi, dc)))
    
    return res


# In[6]:


train_f = creatFeature(train_im)


# In[7]:


valide_f = creatFeature(valide_im)


# In[38]:


from sklearn.feature_selection import VarianceThreshold

def vart(X_tr, X_ts ):
    sel = VarianceThreshold()
    train_ = sel.fit_transform(X_tr)
    p = sel.get_support()
    valide_ = [[x for x,t in zip(im_X, p) if t] for im_X in X_ts]

    return train_, valide_
    


# In[36]:


from sklearn.feature_selection import SelectKBest, f_classif

def sbest(X_tr, y_tr, X_ts , k):
    bestK = SelectKBest(f_classif, k)
    train_ = bestK.fit_transform(X_tr, y_tr)
    p = bestK.get_support()
    valide_ = [[x for x,t in zip(im_X, p) if t] for im_X in X_ts]
    
    return train_, valide_
    


# In[47]:


from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def c_svm(X_tr, y_tr, X_ts, y_ts):
    clf = svm.SVC(gamma='scale')
    clf.fit(X_tr, y_tr) 
    lab = clf.predict(X_ts)
    acc = accuracy_score(y_ts, lab) * 100
    return acc, confusion_matrix(y_ts, lab), lab


# In[48]:


from sklearn.neighbors import KNeighborsClassifier

def knn(X_tr, y_tr, X_ts, y_ts, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_tr, y_tr)
    labels = neigh.predict(X_ts)
    acc = accuracy_score(y_ts, labels) * 100
    return acc, confusion_matrix(y_ts, labels), labels


# In[49]:


from sklearn.naive_bayes import GaussianNB

def gnb(X_tr, y_tr, X_ts, y_ts):
    clf = GaussianNB()
    clf.fit(X_tr, y_tr)
    GaussianNB(priors=None, var_smoothing=1e-09)
    nb_lab = clf.predict(X_ts)
    acc = accuracy_score(y_ts, nb_lab) * 100
    return acc, confusion_matrix(y_ts, nb_lab), nb_lab


# In[50]:


from sklearn import tree

def trd(X_tr, y_tr, X_ts, y_ts):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_tr, y_tr)
    t_lab = clf.predict(X_ts)
    acc = accuracy_score(y_ts, t_lab) * 100
    return acc, confusion_matrix(y_ts, t_lab), t_lab


# In[14]:


def models(X_tr, y_tr, X_ts, y_ts, k):
    print('svm : ')
    r_svm = c_svm(X_tr, y_tr, X_ts, y_ts)
    print(r_svm[0], '\n', r_svm[1])
    
    print('--'*20)
    print('knn : ')
    r_knn = knn(X_tr, y_tr, X_ts, y_ts, k)
    print(r_knn[0], '\n', r_knn[1])
    
    print('--'*20)
    print('gnb : ')
    r_gnb = gnb(X_tr, y_tr, X_ts, y_ts)
    print(r_gnb[0], '\n', r_gnb[1])
    
    print('--'*20)
    print('tree : ')
    r_t = trd(X_tr, y_tr, X_ts, y_ts)
    print(r_t[0], '\n', r_t[1])


# In[144]:


# sbest(15)
train_f2, valide_f2 = vart(train_f, valide_f)
models(train_f2, train.melanoma, valide_f2, valide.melanoma, 10)


# In[159]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_f, train.melanoma, test_size=0.25)
X_train, X_test = vart(X_train, X_test)
c_svmr = c_svm(X_train, y_train, X_test, y_test)

print(c_svmr[0])
c_svmr[1]


# In[160]:


models(X_train, y_train, X_test, y_test, 10)

