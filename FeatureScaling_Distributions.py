#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Reading the dataset
import pandas as pd
import numpy as np
data = pd.read_csv("Pima Indian Diabetes.csv")
data.head()


# In[9]:


# Aplying Standardization to all features
from sklearn.preprocessing import StandardScaler
Y = data.Outcome
X = data.drop("Outcome", axis = 1)
columns = X.columns
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_std = pd.DataFrame(X_std, columns = columns)
X_std.head()


# In[7]:


# Train and Test split of the features
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_std, Y, test_size = 0.15, random_state = 45)


# In[13]:


#Building Logistic Regression model on the Standardized variables
from sklearn.linear_model import LogisticRegression
lr_std = LogisticRegression()
lr_std.fit(x_train, y_train)
y_pred = lr_std.predict(x_test)
print('Accuracy of logistic regression on test set with standardized features: {:.2f}'.format(lr_std.score(x_test, y_test)))


# In[15]:


from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler()
X_norm = norm.fit_transform(X)
X_norm = pd.DataFrame(X_norm, columns = columns)
X_norm.head()


# In[17]:


# Train and Test split of Normalized features
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(X_norm, Y, test_size = 0.15, random_state = 45)


# In[18]:


#Building Logistic Regression model on the Normalized variables
from sklearn.linear_model import LogisticRegression
lr_norm = LogisticRegression()
lr_norm.fit(x1_train, y1_train)
y_pred = lr_norm.predict(x1_test)
print('Accuracy of logistic regression on test set with Normalized features: {:.2f}'.format(lr_norm.score(x1_test, y1_test)))


# In[21]:


# Plotting the histograms of each variable
from matplotlib import pyplot
data.hist(alpha=0.5, figsize=(10, 10))
pyplot.show()


# In[27]:


#Initializing Gaussian and Non-Gaussian features based on distributions
# Standardizing - Gaussian Distribution features
# Normalizing - Non-Gaussian Distribution features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
Standardize_Var = ['BMI','BloodPressure', 'Glucose']
Standardize_transformer = Pipeline(steps=[('standard', StandardScaler())])
Normalize_Var = ['Age','DiabetesPedigreeFunction','Insulin','Pregnancies','SkinThickness']
Normalize_transformer = Pipeline(steps=[('norm', MinMaxScaler())])


# In[30]:


x2_train, x2_test, y2_train, y2_test = train_test_split(X, Y, test_size=0.2)
preprocessor = ColumnTransformer(transformers=
        [('standard', Standardize_transformer, Standardize_Var),
        ('norm', Normalize_transformer, Normalize_Var)])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])
clf.fit(x2_train, y2_train)
print('Accuracy of Logistic Regression model after standardizing Gaussian distributed features and normalizing Non-Gaussian distributed features: {:.2f}'.format(clf.score(x2_test, y2_test)))


# In[ ]:





# In[ ]:




