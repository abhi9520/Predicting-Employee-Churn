
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
import seaborn as sns
os.getcwd()
os.chdir('C:/Users/Mann-A2/Documents/Python Repository')

safa = pd.read_excel("safa.xlsx")

#Exploring Dataset columns
safa.columns
safa.head()

#Missing data
safa.isnull().sum()
safa.isnull().sum().sum()
#dropping columns where all elements are missing
safa = safa.dropna(axis='columns', how = 'all')
safa.isnull().sum().sum()

#dropping Pay Frequency Code as it is hourly for all instances
safa['Pay Frequency Code'].describe()
safa = safa.drop('Pay Frequency Code', axis=1)

#### Visualizations ####
#Distribution summary
safa.hist(figsize=(50,20))
plt.show()

# Find correlations with turnover and sort
safa_corr = safa.drop(["Turnover (1/0)", "Employee Id"], axis=1)
correlations = safa.corr()['Turnover (1/0)'].sort_values()
print('Most Positive Correlations: \n', correlations.tail(12))
print('\nMost Negative Correlations: \n', correlations.head(10))
correlations
# Calculate correlations
corr = safa_corr.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
# Heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr,
            vmax=.5,
            mask=mask,
            # annot=True, fmt='.2f',
            linewidths=.2, cmap="Spectral")


#=====================Feature Engineering================================

#assuming data is collected on 2019/05/30
dataset_date = datetime.datetime(2019,5,30) 

#Experience  = difference in hire date and data collection date
safa['Experience(Years)'] = round((dataset_date - safa['Hire Date'])/np.timedelta64(1,'Y')).astype(int)

#Years left in Company = difference in data collection date and Retirement date 
safa['Years left in Company'] = round((safa['Retirement Date'] - dataset_date)/np.timedelta64(1,'Y')).astype(int)

#Sick days left per year = difference in Sick Days Taken Per Year and Annual Sick Day Allotment
safa['Leaves Remaining'] = safa['Annual Sick Day Allotment'] - safa['Sick Days Taken Per Year']

# Removing Hire Date, Retirement Date and Birth Date
safa = safa.drop(['Hire Date','Retirement Date','Birth Date'], axis=1)

#Binning:
def binning(col, cut_points, labels=None):
  #Define min and max values:
  minval = col.min()
  maxval = col.max()

  #create list by adding min and max to cut_points
  break_points = [minval] + cut_points + [maxval]

  #if no labels provided, use default labels 0 ... (n-1)
  if not labels:
    labels = range(len(cut_points)+1)

  #Binning using cut function of pandas
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin

#Binning age:
cut_points = [30,40,50,60]
labels = ['<=30','31-40','41-50','51-60','60+']
safa['Age_Bin'] = binning(safa['Age'], cut_points, labels)
pd.value_counts(safa['Age_Bin'], sort=False)

#Binning Pay Amount:
cut_points = [30,45,60]
labels = ['low','Medium','High','Very High']
safa['PayScale_Bin'] = binning(safa['Pay Amount'], cut_points, labels)
pd.value_counts(safa['PayScale_Bin'], sort=False)


# Working with Survey Values:
# Removing S6.1 to S6.5 and S6.27 since we dont have much info. (arbitrary answers)
safa = safa.drop(['S6.1', 'S6.2','S6.3','S6.4','S6.5','S6.27'], axis=1)

#Scaling survey results as per their value ranges
safa['S1_results'] = (safa['S1.1'] + safa['S1.2'] + safa['S1.3'] + safa['S1.4'] + safa['S1.5'] + safa['S1.6'])/30

safa['S2_results'] = (safa['S2.1'] + safa['S2.2'] + safa['S2.3'] + safa['S2.4'] + safa['S2.5'] + safa['S2.6'] + safa['S2.7'] + safa['S2.8'] + safa['S2.9'])/54

safa['S3_results'] = (safa['S3.1'] + safa['S3.2'] + safa['S3.3'] + safa['S3.4'] + safa['S3.5'] + safa['S3.6'] + safa['S3.7'] + safa['S3.8'] + safa['S3.9'] + safa['S3.10'] + safa['S3.11'] + safa['S3.12'] + safa['S3.13'] + safa['S3.14'] + safa['S3.15'] + safa['S3.16'] + safa['S3.17'] + safa['S3.18'] + safa['S3.19'] + safa['S3.20'] + safa['S3.21'] + safa['S3.22'] + safa['S3.23'] + safa['S3.24'] + safa['S3.25'] + safa['S3.26'] + safa['S3.27'] + safa['S3.28'] + safa['S3.29'] + safa['S3.30'] + safa['S3.31'] + safa['S3.32'] + safa['S3.33'] + safa['S3.34'] + safa['S3.35'] + safa['S3.36'])/216

safa['S5_results'] = (safa['S5.1'] + safa['S5.2'] + safa['S5.3'] + safa['S5.4'] + safa['S5.5'] + safa['S5.6'] + safa['S5.7'])/35

safa['S6_results'] = (safa['S6.6'] + safa['S6.7'] + safa['S6.8'] + safa['S6.9'] + safa['S6.10'] + safa['S6.11'] + safa['S6.12'] + safa['S6.13'] + safa['S6.14'] + safa['S6.15'] + safa['S6.16'] + safa['S6.17'] + safa['S6.18'] + safa['S6.19'] + safa['S6.20'] + safa['S6.21'] + safa['S6.22'] + safa['S6.23'] + safa['S6.24'] + safa['S6.25'] + safa['S6.26'])/147

#dropping columns that are not required now.
safa = safa.drop(['Age', 'Pay Amount','Annual Sick Day Allotment', 'Sick Days Taken Per Year', 'S1.1', 'S1.2', 'S1.3', 'S1.4', 'S1.5', 'S1.6', 'S2.1', 'S2.2', 'S2.3', 'S2.4', 'S2.5', 'S2.6', 'S2.7', 'S2.8', 'S2.9', 'S3.1', 'S3.2', 'S3.3', 'S3.4', 'S3.5', 'S3.6', 'S3.7', 'S3.8', 'S3.9', 'S3.10', 'S3.11', 'S3.12', 'S3.13', 'S3.14', 'S3.15', 'S3.16', 'S3.17', 'S3.18', 'S3.19', 'S3.20', 'S3.21', 'S3.22', 'S3.23', 'S3.24', 'S3.25', 'S3.26', 'S3.27', 'S3.28', 'S3.29', 'S3.30', 'S3.31', 'S3.32', 'S3.33', 'S3.34', 'S3.35', 'S3.36', 'S5.1', 'S5.2', 'S5.3', 'S5.4', 'S5.5', 'S5.6', 'S5.7', 'S6.6', 'S6.7', 'S6.8', 'S6.9', 'S6.10', 'S6.11', 'S6.12', 'S6.13', 'S6.14', 'S6.15', 'S6.16', 'S6.17', 'S6.18', 'S6.19', 'S6.20', 'S6.21', 'S6.22', 'S6.23', 'S6.24', 'S6.25', 'S6.26'], axis=1)


#rearranging the columns
df1 = safa.pop('Turnover (1/0)')
safa['Turnover (1/0)'] = df1 

safa.to_excel (r'safa_updated Feature Engg. v2.1.xlsx', index = None, header=True)

#=================Feature Scaling ========================
# Scaling the data (Standardization - Z Score normalization)
scaler = StandardScaler()
features = ['Number of Dependents', '1. Withdraw Cognitions', '2. Organizational Commitment', '3. Job Satisfaction', '4. Rewards Offered Beyond Pay (Comes from 3.13, 3.22, 3.23)', '5. Work Stress', 'Experience(Years)', 'Years left in Company', 'Leaves Remaining']
safa[features] = scaler.fit_transform(safa[features])
safa.head()

safa.to_excel (r'safa_updated Feature Scaled v2.2.xlsx', index = None, header=True)

# Normalization (Box Cox transforamtion)
#does not work if data has values <= 0

#safa = pd.read_excel("unscaled.xlsx")
#features = ['1. Withdraw Cognitions']
#safa[features] = power_transform(safa[features], method='box-cox')
#safa.head()
#
#safa.to_excel (r'safa_updated Box-Cox.xlsx', index = None, header=True)
#

#min-max scaling
#safa = pd.read_excel("unscaled.xlsx")
#scaler = MinMaxScaler()
#features = ['Number of Dependents', '1. Withdraw Cognitions', '2. Organizational Commitment', '3. Job Satisfaction', '4. Rewards Offered Beyond Pay (Comes from 3.13, 3.22, 3.23)', '5. Work Stress', 'Experience(Years)', 'Years left in Company', 'Leaves Remaining']
#safa[features] = scaler.fit_transform(safa[features])
#safa.head()
#
#safa.to_excel (r'safa_updated Min Max Scaled.xlsx', index = None, header=True)

#=================Creating dummy variables ========================
#converting categorical variables to dummy variables
dummy_columns = ['NOC Code', 'Employee Group', 'Employee Status', 'Gender', 'Generation', 'Department', 'Age_Bin', 'PayScale_Bin']

safa_dummies = pd.get_dummies(safa[dummy_columns])
safa = safa.join(safa_dummies)

#dropping categorical columns
safa = safa.drop(dummy_columns, axis=1)

#rearranging the columns
df1 = safa.pop('Turnover (1/0)')
safa['Turnover (1/0)'] = df1 

safa.to_excel (r'[Final] Safa_Cleaned.xlsx', index = None, header=True)



#=============================================================================
#Upsampling Code
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:44:14 2019

@author: Mann-A2
"""

import pandas as pd
import os
import numpy as np

os.getcwd()
os.chdir('C:/Users/Mann-A2/Documents/Python Repository')

safa = pd.read_excel("[Final] Safa_Cleaned.xlsx")

safa.head()
safa['Turnover (1/0)'].value_counts()

from sklearn.utils import resample
df_majority = safa[safa['Turnover (1/0)'] == 0]
df_minority = safa[safa['Turnover (1/0)'] == 1] 

# Upsample minority class   # sample with replacement   # to match majority class
df_minority_upsampled = resample(df_minority, replace=True, n_samples=3543, random_state=123)
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled['Turnover (1/0)'].value_counts()

df_upsampled.to_excel (r'safa_updated upsample.xlsx', index = None, header=True)


#=============================

safa = pd.read_excel("safa_updated upsample.xlsx")

from sklearn.model_selection import train_test_split

X=safa[['Employee Id', 'Number of Dependents', '1. Withdraw Cognitions', '2. Organizational Commitment', '3. Job Satisfaction', '4. Rewards Offered Beyond Pay (Comes from 3.13, 3.22, 3.23)', '5. Work Stress', 'Experience(Years)', 'Years left in Company', 'Leaves Remaining', 'S1_results', 'S2_results', 'S3_results', 'S5_results', 'S6_results', 'NOC Code_0', 'NOC Code_A', 'NOC Code_B', 'NOC Code_C', 'NOC Code_D', 'Employee Group_Executive', 'Employee Group_Management', 'Employee Group_Staff', 'Employee Status_C', 'Employee Status_FT', 'Employee Status_PT', 'Gender_F', 'Gender_M', 'Generation_Baby Boomer', 'Generation_Gen X', 'Generation_Millennials', 'Department_A', 'Department_B', 'Department_C', 'Department_D', 'Age_Bin_<=30', 'Age_Bin_31-40', 'Age_Bin_41-50', 'Age_Bin_51-60', 'Age_Bin_60+', 'PayScale_Bin_low', 'PayScale_Bin_Medium', 'PayScale_Bin_High', 'PayScale_Bin_Very High']]  # Features
y=safa['Turnover (1/0)']  # Labels

# Split dataset into training set, test set and validation set 60%, 20%, 20%
X_train, X_val, X_test = np.split(X.sample(frac=1), [int(.6*len(X)), int(.8*len(X))])

y_train, y_val, y_test = np.split(y.sample(frac=1), [int(.6*len(y)), int(.8*len(y))])

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=28)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("f1 score:",metrics.f1_score(y_test, y_pred))
print("recall score:",metrics.recall_score(y_test, y_pred))
print("roc auc score:",metrics.roc_auc_score(y_val, y_pred_val))


#Train the model using the training sets y_pred=clf.predict(X_test)
y_pred_val=clf.predict(X_val)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_val, y_pred_val))
print("f1 score:",metrics.f1_score(y_val, y_pred_val))
print("recall score:",metrics.recall_score(y_val, y_pred_val))





#==============================================================================
#Model Building

import pandas as pd
import os
import numpy as np

os.getcwd()
os.chdir('C:/Users/Mann-A2/Documents/Python Repository')

safa = pd.read_excel("[Final] Safa_Cleaned without binning.xlsx")

from sklearn.model_selection import train_test_split

X=safa.iloc[:,:44] #Selecting all X features
y=safa['Turnover (1/0)']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=123)

# Performing SMOTE on train dataset
from imblearn.over_sampling import SMOTE
X_train,y_train = SMOTE().fit_resample(X_train,y_train)
X_train.shape
y_train.shape
        
#=====Basic Random Forest===========================

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=28)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

#=====Performance Measures=================================
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy Score:",metrics.balanced_accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))
print("Avg Precision Score:",metrics.average_precision_score(y_test, y_pred))
print("Recall score:",metrics.recall_score(y_test, y_pred))

print("f1 score:",metrics.f1_score(y_test, y_pred))
print("Log Loss:",metrics.log_loss(y_test, y_pred))

print("ROC Curve:",metrics.roc_curve(y_test, y_pred))

print("ROC_AUC_Score:",metrics.roc_auc_score(y_test, y_pred))

cm1 = metrics.confusion_matrix(y_test, y_pred)
print("confusion matrix:\n", cm1)


#PLotting confusion matrix
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual Turnover')
    plt.xlabel('Predicted Turnover')

np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm1, classes=[0,1],
                      title='Confusion matrix')


#Plotting ROC Curve

plt.figure(figsize=(8, 8))
plt.title("ROC Curve", fontsize=18)
plt.xlabel("False Positive Rate (100-Specificity)", fontsize=14)
plt.ylabel("True Positive Rate (Sensitivity", fontsize=14)
plt.tick_params(labelsize=12);

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)

plt.show()

#==============FEATURE SELECTION====================================
#Checking feature importance
feature_importances = pd.DataFrame(clf.feature_importances_,
                                   index = X.columns,
                                   columns=['importance']).sort_values('importance',
                                           ascending=False)

print(feature_importances)

#Feature Selection using Randm Forest

from sklearn.feature_selection import SelectFromModel

clf = RandomForestClassifier(n_estimators = 100) # Create RF Model parameters
sel = SelectFromModel(clf) #Ask RF Model to select best features too
sel.fit(X_train, y_train) # fit on training data

#features whose importance is > mean importance are kept
selected_feat= X.columns[(sel.get_support())]
len(selected_feat)
print(selected_feat) #see which features are important and kept


#Feature selection using Threshold
# features that have an importance of more than 0.15
sfm = SelectFromModel(clf, threshold=0.15)
# Train the selector
sfm.fit(X_train, y_train)

#Selected features
selected_feat= X.columns[(sfm.get_support())]
len(selected_feat)
print(selected_feat) 

#Feature selection using RFE 
# Recursive feature elimination with built-in CV selection of the best number of features
from sklearn.feature_selection import RFE
selector = RFE(clf, n_features_to_select=None, step=1) #none selects half features
selector = selector.fit(X_train, y_train)

#Selected features
selected_feat= X.columns[(selector.get_support())]
len(selected_feat)
print(selected_feat)

#==================================================

#Hyperparameter tuning

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# n_estimators : The number of trees in the forest. default = 100
#bootstrap=True(default) samples are drawn with replacement
# criterion : default=”gini”, "entropy" to measure the quality of a split.
#max_depth : The maximum depth of the tree :def=None : expanded till all leaves are pure
# random_state = 123 : sets seed
# class_weight : for class sensitive learning
# If not given, all classes are supposed to have weight one. Def: None
# Best:  class_weight = "balanced" : classes are automatically weighted
# inversely proportional to how frequently they appear in the data. 
# {0:.01, 1: .99} : Now, class 0 has weight 1 and class 1 has weight 2.

clf=RandomForestClassifier(n_estimators=100, criterion ="entropy", max_depth = 10, 
                           class_weight = "balanced", random_state=123)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("f1 score:",metrics.f1_score(y_test, y_pred))
print("recall score:",metrics.recall_score(y_test, y_pred))
cm1 = metrics.confusion_matrix(y_test, y_pred)
print("confusion matrix:\n", cm1)


# Random Hyperparameter Grid, RandomizedSearchCV
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Random Search Training

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5,
                               verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train,y_train)

rf_random.best_params_

y_pred=rf_random.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("f1 score:",metrics.f1_score(y_test, y_pred))
print("recall score:",metrics.recall_score(y_test, y_pred))
cm1 = metrics.confusion_matrix(y_test, y_pred)
print("confusion matrix:\n", cm1)


# Grid Search with Cross Validation

from sklearn.model_selection import GridSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 400, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 8, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(param_grid)

# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search model
grid_search.fit(X_train,y_train)

grid_search.best_params_

y_pred=grid_search.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy Score:",metrics.balanced_accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))
print("Avg Precision Score:",metrics.average_precision_score(y_test, y_pred))
print("Recall score:",metrics.recall_score(y_test, y_pred))

print("f1 score:",metrics.f1_score(y_test, y_pred))
print("Log Loss:",metrics.log_loss(y_test, y_pred))

print("ROC Curve:",metrics.roc_curve(y_test, y_pred))

print("ROC_AUC_Score:",metrics.roc_auc_score(y_test, y_pred))

cm1 = metrics.confusion_matrix(y_test, y_pred)
print("confusion matrix:\n", cm1)


#==========Gradient Boosting=============================
from sklearn.ensemble import GradientBoostingClassifier

# fit model no training data
gboost = GradientBoostingClassifier(random_state=123,n_estimators=100 )

# Fit the xg boost model
gboost.fit(X_train,y_train)

y_pred=gboost.predict(X_test)

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("f1 score:",metrics.f1_score(y_test, y_pred))
print("recall score:",metrics.recall_score(y_test, y_pred))
cm1 = metrics.confusion_matrix(y_test, y_pred)
print("confusion matrix:\n", cm1)


gboost.feature_importances_

#Checking feature importance for gboost
feature_importances = pd.DataFrame(gboost.feature_importances_,
                                   index = X.columns,
                                   columns=['importance']).sort_values('importance',
                                           ascending=False)

print(feature_importances)

feat_importances = pd.Series(gboost.feature_importances_, index=list(X.columns.values))
feat_importances.nlargest(10).plot(kind='barh')


#==========Extreme Gradient Boosting=============================
# pip install xgboost
from xgboost import XGBClassifier

#we have to convert into arrays first due to a bug.
X_train = np.array(X_train)
X_test = np.array(X_test)

# fit model no training data
xgboost = XGBClassifier(max_depth=7,
                           min_child_weight=1,
                           learning_rate=0.1,
                           n_estimators=500,
                           silent=True,
                           objective='binary:logistic',
                           gamma=0,
                           max_delta_step=0,
                           subsample=1,
                           colsample_bytree=1,
                           colsample_bylevel=1,
                           reg_alpha=0,
                           reg_lambda=0,
                           scale_pos_weight=1,
                           seed=1,
                           missing=None)
# Fit the xg boost model
xgboost.fit(X_train,y_train)

y_pred=xgboost.predict(X_test)

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("f1 score:",metrics.f1_score(y_test, y_pred))
print("recall score:",metrics.recall_score(y_test, y_pred))
cm1 = metrics.confusion_matrix(y_test, y_pred)
print("confusion matrix:\n", cm1)

#Checking feature importance for gboost
feature_importances = pd.DataFrame(xgboost.feature_importances_,
                                   index = X.columns,
                                   columns=['importance']).sort_values('importance',
                                           ascending=False)

print(feature_importances)

#=======================================



