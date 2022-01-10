# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 19:12:59 2021

@author: rauta
"""

# Importing Libraries
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report

# Importing dataset 
dataset = pd.read_csv(r'C:\Users\rauta\Data Science\Project\train.csv')


# Creating copy of a dataset
data = dataset.copy(deep=True)

#####################################################################################################

# Analysing Data

#####################################################################################################

#checking top 5 values of the dataset
data.head()

#to check the infomartion of the dataset
data.info()

#to check is null values are present
data.isnull().sum()

# to check detailed numerical values of quantitative variables
num_data = data.describe()
# to check detailed numerical values of qualitative variables
cat_data = data.describe(include="O")

# to remove unwanted columns
col = ['PassengerId','Name','Ticket','Cabin']
data = data.drop(columns=col,axis=1)

#creating copy of data
data1 = data.copy(deep=True)
#data = data.drop(columns='Cabin',axis=1)
data.info()


# to check null values in embarked column and replacing it with most occuring index 
data['Embarked'].describe()
data['Embarked'].value_counts()
data['Embarked'].value_counts().index[0]
#to replace null values with most occuring index
data['Embarked'].fillna(data['Embarked'].value_counts().index[0],inplace=True)

#to check mean of age column
data['Age'].mean()
#to replace null value with mean value
data['Age'].fillna(data['Age'].mean(),inplace=True)
#converting age column to int
data['Age'] = data['Age'].astype('int64')

#function to categorize age with age group
def age_group(data):
    if (data['Age']>0) & (data['Age']<=2):
        return "Baby"
    elif (data['Age']>=3) & (data['Age']<=17):
        return "Child"
    elif (data['Age']>=18) & (data['Age']<=65):
        return "Adult"
    elif (data['Age']>=65):
        return "Elderly"

data['AgeGroup'] = data.apply(lambda ab:age_group(ab),axis=1)

"""
Alternate we can use cut function

agegroup = pd.cut(data.Age,bins=[0,2,17,65,99],labels=['Baby','Child','Adult','Elderly'])
data.insert(4,'AgeGroup',agegroup)

"""

data.info()

#####################################################################################################

# Visualising data

#####################################################################################################

#to plot countplot graph for survived column
sns.countplot('Survived', data=data)

#seperating few of the columns and adding index to them
feature = ['Pclass','Sex','SibSp','Parch','Embarked','AgeGroup']
list(enumerate(feature))
#plotting countplot for all the columns
plt.figure(figsize=(10,10))
for i in enumerate(feature):
    plt.subplot(3, 3, i[0]+1)
    sns.countplot(x=i[1], data=data,hue='Survived')
    
#to get matrix of surival rate by sex
data.groupby('Sex')[['Survived']].mean()

#to get matrix of survival rate wrt sex and class
data.pivot_table('Survived', index='Sex' , columns='Pclass')
#to plot graph of survival rate by sex and class
data.pivot_table('Survived', index='Sex' , columns='Pclass').plot()

#to plot bar graph for all the columns
plt.figure(figsize=(10,10))
for i in enumerate(feature):
    plt.subplot(3, 3, i[0]+1)
    sns.barplot(x=i[1], y='Survived',data=data)

#to get matrix of survival rate wrt sex, age and clas
age = pd.cut(data['Age'],[0,18,45,99])
data.pivot_table('Survived', ['Sex',age] ,'Pclass')
data.pivot_table('Survived', ['Sex',age] ,'Pclass').plot()

#to plot scatter plot for fare vs pclass
plt.scatter(data['Fare'], data['Pclass'])
plt.xlabel('Fare')
plt.ylabel('Class')

#to plot graph for numerical columns i.e age and fare 
sns.distplot(data['Age'])  
sns.distplot(data['Fare'])  

#to create box plot for numerical columns i.e age and column
feature1 = ['Age','Fare']
list(enumerate(feature1))

for i in enumerate(feature1):
    plt.subplot(1,2,i[0]+1)
    sns.boxplot(y=i[1],data = data)

"""
#to create pie chart for survival
import plotly.express as px

fig = px.pie(data,names='Survived',color='Survived',
             color_discrete_map={'0':'royalblue',
                                 '1':'cyan'})
plt.plot(fig)
plt.show()
"""

#####################################################################################################


#To seperate qualitative and quatitaive variables
data.select_dtypes(include=[np.number]).columns.tolist()
Quant_var = data[data.select_dtypes(include=[np.number]).columns.tolist()]

Qual_var = data[data.select_dtypes(include=['object']).columns.tolist()] 

#to create dummy variables for categorical data
from sklearn.preprocessing import LabelEncoder
Qual_var = Qual_var.apply(LabelEncoder().fit_transform)  

#concatinating the complete data
finaldata = pd.concat([Quant_var,Qual_var],axis=1)
finaldata.info()
#creating copy of final data
final_data_copy = finaldata.copy(deep=True)

#seperating dependent and independent variables
X = finaldata.drop(['Survived'],axis=1)
Y = finaldata['Survived']

#Splitting data into Train and Test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, train_size=0.80 , random_state=1)

"""
#to scale independent variables to make data good for the model
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.fit_transform(X_test)

"""

#####################################################################################################

#Building Models
 
#####################################################################################################
   
#Logistic Regression
from sklearn.linear_model import LogisticRegression
#lr = LogisticRegression(random_state=1)
#lr.fit(X_train, Y_train)

#KNN
from sklearn.neighbors import KNeighborsClassifier
#to check best suitable k value for the data
error = []
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train,Y_train)
    pred_knn = knn.predict(X_test)
    error.append(np.mean(pred_knn != Y_test))
    
plt.figure(figsize=(10,10))
plt.plot(range(1,30),error,color='red',linestyle='dashed',marker='o')
plt.xlabel('K value')
plt.ylabel('Mean Error')

#knn = KNeighborsClassifier(n_neighbors = 7 , metric = 'minkowski',p=2)  #p values is to calculate distance , we are using euclidean distance hance p=2
#knn.fit(X_train, Y_train)


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
#to check best suitable tree depth size for the data
acc = []
for i in range(1,10):
    tree = DecisionTreeClassifier(max_depth= i, random_state=1 )
    tree.fit(X_train,Y_train)
    pred_tree = tree.predict(X_test)
    score_tree = accuracy_score(Y_test,pred_tree)
    acc.append(score_tree)

plt.figure(figsize=(10,10))
plt.plot(range(1,10),acc,color='red',linestyle='dashed',marker='o')
plt.xlabel('predicted')
plt.ylabel('score')    

#dt = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=1)
#dt.fit(X_train,Y_train)
    
#Random Forest
from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(n_estimators=10 , criterion='entropy',random_state=1)
#rf.fit(X_train,Y_train)

#creating function for the model bulding 
def models(X_train,Y_train):
    
    #LInear Regression
    lr = LogisticRegression(random_state=1)
    lr.fit(X_train, Y_train)
    
    #KNN
    knn = KNeighborsClassifier(n_neighbors = 19 , metric = 'minkowski', p=2)  #p values is to calculate distance , we are using euclidean distance hance p=2
    knn.fit(X_train, Y_train)
    
    #Decision Tree
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=1)
    dt.fit(X_train,Y_train)
    
    #Random Forest
    rf = RandomForestClassifier(n_estimators=10 , criterion='entropy',random_state=1)
    rf.fit(X_train,Y_train)
    
    #printing acuuracy score of the models on the train data
    print('[0] Linear Regression :' , lr.score(X_train,Y_train))
    print('[1] K Nearest Neighbors :' , knn.score(X_train,Y_train))
    print('[2] Decision Tree :' , dt.score(X_train,Y_train))
    print('[3] Random Forest :' , rf.score(X_train,Y_train))
    
    return lr,knn,dt,rf


#traing models 
train_models = models(X_train, Y_train)

#####################################################################################################

#Testing models on test data
for i in range ( len(train_models) ):
    #creating confusion matrix for all the models
    cm = confusion_matrix(Y_test, train_models[i].predict(X_test))
    
    TN, FP, FN, TP = confusion_matrix(Y_test, train_models[i].predict(X_test)).ravel()
    
    test_score = (TP + TN) / (TP + TN + FP + FN)
    #printing accuracy score of all the models on test data
    print(cm)
    print('Model[{}] Testing Accuracy : "{}"'.format(i, test_score))
    print()


#####################################################################################################
# to check models one by one
#####################################################################################################


#to print predicted output of any model
pred = train_models[2].predict(X_test)
print(pred)

print()

#actual values
print(Y_test)

    
#####################################################################################################
# new user survival predication
#####################################################################################################


pclass = int(input("Enter your passenger class (1st: 1, 2nd: 2, 3rd: 3): "))
sex = int(input("Enter your Sex ( Male:1, Female:0) : "))
yourage = int(input("Enter your Age: "))
sibsp = int(input("Enter sib / spouse: "))
parch = int(input("Enter parent / child: "))
fare = float(input("Enter your fare: "))
embark = int(input("Enter your embark(C:0, Q:1, S:3) : "))
agegroup = int(input("Enter your Age Group(0-2: 0, 3-17: 1, 18-65: 2, 66+: 3): "))

survival = [[pclass,sex,yourage,sibsp,parch,fare,embark,agegroup]]

result = train_models[3].predict(survival)
print(result)

if result==0:
    print("You will not survive")

else:
    print("You will survive")
    
#####################################################################################################