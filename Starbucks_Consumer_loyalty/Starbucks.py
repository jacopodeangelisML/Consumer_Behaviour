# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:32:38 2020

@author: Iacopo
"""

#Import libraries
import pandas as pd
import numpy as np
import seaborn as sns

#Import dataset
df = pd.read_csv('Dataset_clean.csv')

#Retrieving data general Information
df.info()
describe = df.describe() #no missing values to signal

#Data distribution --> ok, already checked on Kraggle description
##FEATURES ENGINEERING
#Replacing maximum value "5" of variable "method", which is out of variable range, with the median==1
df.set_value(91,'method',1)
#Removing non-discriminative variables
df.drop(['itemPurchaseCoffee','itempurchaseCold','itemPurchasePastries',
         'itemPurchaseJuices','itemPurchaseSandwiches','itemPurchaseOthers',
         'promoMethodApp','promoMethodSoc','promoMethodEmail','promoMethodDeal',
         'promoMethodFriend','promoMethodDisplay','promoMethodBillboard',
         'promoMethodOthers'], axis=1, inplace=True)
#Dummy coding of nominal variables with k levels
status = pd.get_dummies(df['status'],drop_first=True)
status = status.rename(columns={0:'Student',1:'Self-Employed',2:'Employed',3:'Housewife'})
method = pd.get_dummies(df['method'],drop_first=True)
method = method.rename(columns={0:'Dine In',1:'Drive-thru',2:'Take away',3:'Never'})
#Appending status and method and removing old variables
df.drop('status', axis=1, inplace=True)
df.drop('method', axis=1, inplace=True)
df_1 = pd.concat([df,status, method], axis=1)

#SEPARATING X set and y
X = df_1.drop('loyal', axis=1)
y = df_1['loyal']

##MANN-WHITNEY TESTS
from scipy.stats import mannwhitneyu
#Age
loyal = df.where(df.loyal== 1).dropna()['age']
noloyal = df.where(df.loyal== 0).dropna()['age']
print(mannwhitneyu(loyal, noloyal))
#Loyal
loyal = df.where(df.loyal== 1).dropna()['income']
noloyal = df.where(df.loyal== 0).dropna()['income']
print(mannwhitneyu(loyal, noloyal))
#Visits
loyal = df.where(df.loyal== 1).dropna()['visitNo']
noloyal = df.where(df.loyal== 0).dropna()['visitNo']
print(mannwhitneyu(loyal, noloyal))
df.groupby('loyal').mean()['visitNo']
sns.barplot(x=df['loyal'], y=df['visitNo'], data=df)
#Time Spend
loyal = df.where(df.loyal== 1).dropna()['timeSpend']
noloyal = df.where(df.loyal== 0).dropna()['timeSpend']
print(mannwhitneyu(loyal, noloyal))
#Location
loyal = df.where(df.loyal== 1).dropna()['location']
noloyal = df.where(df.loyal== 0).dropna()['location']
print(mannwhitneyu(loyal, noloyal))
#Spend Purchase
loyal = df.where(df.loyal== 1).dropna()['spendPurchase']
noloyal = df.where(df.loyal== 0).dropna()['spendPurchase']
print(mannwhitneyu(loyal, noloyal))
#ProductRate
loyal = df.where(df.loyal== 1).dropna()['productRate']
noloyal = df.where(df.loyal== 0).dropna()['productRate']
print(mannwhitneyu(loyal, noloyal))
sns.boxplot(x=df['loyal'], y=df['productRate'], data=df)
#priceRate
loyal = df.where(df.loyal== 1).dropna()['priceRate']
noloyal = df.where(df.loyal== 0).dropna()['priceRate']
print(mannwhitneyu(loyal, noloyal))
sns.boxplot(x=df['loyal'], y=df['priceRate'], data=df)
#ambianceRate
loyal = df.where(df.loyal== 1).dropna()['ambianceRate']
noloyal = df.where(df.loyal== 0).dropna()['ambianceRate']
print(mannwhitneyu(loyal, noloyal))
sns.boxplot(x=df['loyal'], y=df['ambianceRate'], data=df)
#wifiRate
loyal = df.where(df.loyal== 1).dropna()['wifiRate']
noloyal = df.where(df.loyal== 0).dropna()['wifiRate']
print(mannwhitneyu(loyal, noloyal))
#serviceRate
loyal = df.where(df.loyal== 1).dropna()['serviceRate']
noloyal = df.where(df.loyal== 0).dropna()['serviceRate']
print(mannwhitneyu(loyal, noloyal))
#chooseRate
loyal = df.where(df.loyal== 1).dropna()['chooseRate']
noloyal = df.where(df.loyal== 0).dropna()['chooseRate']
print(mannwhitneyu(loyal, noloyal))
sns.boxplot(x=df['loyal'], y=df['chooseRate'], data=df)

##CHI-SQUARE CORRELATION
import scipy.stats as stats
#Gender
crosstab = pd.crosstab(df["gender"], df["loyal"])
stats.chi2_contingency(crosstab)
#Status
crosstab = pd.crosstab(df["membershipCard"], df["loyal"])
stats.chi2_contingency(crosstab)
sns.countplot(x=df["membershipCard"], hue= df["loyal"])
#SelfEmployed
crosstab = pd.crosstab(df_1["Self-Employed"], df["loyal"])
stats.chi2_contingency(crosstab)
#Employed
crosstab = pd.crosstab(df_1["Employed"], df["loyal"])
stats.chi2_contingency(crosstab)
#Housewife
crosstab = pd.crosstab(df_1["Housewife"], df["loyal"])
stats.chi2_contingency(crosstab)
#Drive-thru
crosstab = pd.crosstab(df_1["Drive-thru"], df["loyal"])
stats.chi2_contingency(crosstab)
#Take away
crosstab = pd.crosstab(df_1["Take away"], df["loyal"])
stats.chi2_contingency(crosstab)

##INPUT SELECTION
#Lasso method
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
lr_selector = SelectFromModel(LogisticRegression(penalty='l1'), max_features=10)
lr_selector.fit(X,y)
lr_support = lr_selector.get_support()
lr_feature = X.loc[:,lr_support].columns.tolist()
print(str(len(lr_feature)), 'selected features')
#Select relevant inputs
X = X.loc[:,['income','visitNo','timeSpend','membershipCard','spendPurchase','priceRate','ambianceRate','serviceRate','chooseRate',
             'Self-Employed']]

#Trainining test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5, random_state = 42)

#SVM
from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)

#Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
print(confusion_matrix(y_test,y_pred)) #Confusion mattrix
print(classification_report(y_test,y_pred))

##REBALANCING DATASET BASED ON OUTCOME LEVELS
from imblearn.over_sampling import SMOTE #Importing SMOTE module to balance the dataset
sm = SMOTE(sampling_strategy='minority', random_state=7) #Resample the minority class
X, y = sm.fit_sample(df_1.drop('loyal', axis=1), df_1['loyal']) #Fit the model to generate the data, and create X and Y
X = pd.DataFrame(X)  

#Lasso method
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
lr_selector = SelectFromModel(LogisticRegression(penalty='l1'), max_features=10)
lr_selector.fit(X,y)
lr_support = lr_selector.get_support()
lr_feature = X.loc[:,lr_support].columns.tolist()
print(str(len(lr_feature)), 'selected features')
X = X.iloc[:,[4,5,8,9,10,14,15,16,17,20]]
#'visitNo','spendPurchase','timeSpend','productRate','priceRate','serviceRate',chooseRate','Self-Employed','Employed','Take away
#Trainining test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5, random_state = 42)

#SVM
from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)

#Evaluation metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
print(confusion_matrix(y_test,y_pred)) #Confusion mattrix
print(classification_report(y_test,y_pred)) #Precision, Recall, Accuracy, F1-score
print(roc_auc_score(y_test, y_pred)) #Area under the curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)') #Plotting roc curve
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

##10-CROSS VALIDATION TO DEAL WITH OVERFITTING
from sklearn.model_selection import cross_validate #Importing cross_validate module (it allows to compute multiple metrics)
from sklearn.model_selection import cross_val_score #Importing cross_val_score (it allows to obtain accuracy)
scores = cross_validate(svm, X, y, cv=10,scoring= ['precision_macro','recall_macro','f1_macro']) #Running 10-fold cross validation (evaluation: precision, recall, f1)
score = cross_val_score(svm, X, y, cv=10) #Running 10-fold cross validation (accuracy)
iterations = [1,2,3,4,5,6,7,8,9,10] 
recall = list(scores['test_recall_macro'])
precision = list(scores['test_precision_macro'])
f1 = list(scores['test_f1_macro'])
accuracy = list(score)
summary_table = pd.DataFrame(list(zip(iterations,recall,precision,f1,accuracy)),columns=['iterations','recall','precision','f1','accuracy'])
fig= sns.lineplot(x=summary_table['iterations'], y=summary_table['accuracy'])
summary_table['f1'].mean()

#Features importance
features_name = ['visitNo','spendPurchase','timeSpend','productRate','priceRate','serviceRate',
                 'chooseRate','Self-Employed','Employed','Take away']
svm.coef_
tfi = pd.DataFrame({'Features':['number of visits', 'Spent Purchased', 'Time in stores', 'Product Rate',
                                'Price Rate', 'Service Rate', 'Choose Rate', 'Self-Employed', 'Employed',
                                'Take Away'], 'importance': [-0.372, 0.166, 1.415, 0.483, 0.816, -0.378, 
                                0.433, 0.057, 0.750, 0.878]})
tfi_sort = tfi.sort_values(by =['importance'])
sns.barplot(x=(tfi_sort['importance']),y=tfi_sort['Features'])













