import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
original_train = train.copy()
full_data = [train,test]

train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x)==float else 1)
test['Has_Cabin'] = test['Cabin'].apply(lambda x: 0 if type(x)==float else 1)

for dataset in full_data:
    dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch'] + 1
    
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['Family_Size']==1, 'IsAlone'] = 1
        
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    
for dataset in full_data:
    avg_age = dataset['Age'].mean()
    avg_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    random_list = np.random.randint(avg_age-avg_std,avg_age+avg_std,age_null_count)
    dataset.loc[np.isnan(dataset['Age']),'Age'] = random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                            'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
for dataset in full_data:
    sex_map = {'female':0, 'male':1}
    dataset['Sex'] = dataset['Sex'].map(sex_map).astype(int)
    
    emb_map = {'S':0, 'C':1, 'Q':2}
    dataset['Embarked'] = dataset['Embarked'].map(emb_map).astype(int)
    
    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']  = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    dataset.loc[ dataset['Age'] <= 16, 'Age']  = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'];
    

drop_elements = ['PassengerId','Name','Ticket','Cabin','SibSp']
train = train.drop(drop_elements,axis=1)
test = test.drop(drop_elements,axis=1)

cv = KFold(n_splits=10)
accuracies = list()
max_attributes = len(list(test))
depth_range = range(1,max_attributes+1)

for depth in depth_range:
    fold_acc = []
    tree_model = tree.DecisionTreeClassifier(max_depth = depth)
    
    for train_fold,valid_fold in cv.split(train):
        f_train = train.loc[train_fold]
        f_valid = train.loc[valid_fold]
        
        model = tree_model.fit(X = f_train.drop(['Survived'],axis=1),
                               y = f_train['Survived'])
        
        valid_acc = model.score(X = f_valid.drop(['Survived'],axis=1),
                                y = f_valid['Survived'])
        
        fold_acc.append(valid_acc)
        
    avg = sum(fold_acc)/len(fold_acc)
    accuracies.append(avg)
    

x_train = train.drop(['Survived'],axis=1).values
y_train = train['Survived']
x_test = test.values

best_acc = 0
best_depth = 0
for i in range(len(accuracies)):
    if(accuracies[i]>best_acc):
        best_acc = accuracies[i]
        best_depth = i+1
        
decision_tree = tree.DecisionTreeClassifier(max_depth=best_depth)
decision_tree.fit(x_train,y_train)

y_pred = decision_tree.predict(x_test)
submission = pd.DataFrame({"PassengerId": PassengerId,
        "Survived": y_pred})
submission.to_csv('submission.csv', index=False)

    
        



    
