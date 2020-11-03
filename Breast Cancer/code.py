import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

full_data = pd.read_csv('data.csv')
full_data.drop(['Unnamed: 32','id'],axis=1,inplace=True)
full_data['diagnosis'] = full_data['diagnosis'].apply(lambda x: 0 if x=='B'else 1)

x = full_data.drop(['diagnosis'],axis=1)
y = full_data['diagnosis'].values
x = x.to_numpy()

x = (x-np.min(x))/(np.max(x)-np.min(x))

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=42)


num_features = x_train.shape[1]
theta = np.full((num_features),0.01)
bias = 0.0
lower_range = 5
upper_range = 200
step = 1
lr_rate_range = range(lower_range,upper_range,step)
tot_lr = (upper_range-lower_range)//step + 1
accuracies = []
num_splits = 8
num_each = x_train.shape[0]//num_splits
num_epochs = 500

for lr_rate in lr_rate_range:
    fold_acc = []
    for j in range(num_splits):
        temp_theta = theta.copy()
        temp_bias = 0.0
        xf_valid = x_train[j*num_each:(j+1)*num_each]
        yf_valid = y_train[j*num_each:(j+1)*num_each]
        xf_train = np.concatenate((x_train[0:j*num_each],x_train[(j+1)*num_each:]),axis=0)
        yf_train = np.concatenate((y_train[0:j*num_each],y_train[(j+1)*num_each:]),axis=0)
        for itr in range(num_epochs):
            z = xf_train.dot(temp_theta) + temp_bias
            h = 1/(1+np.exp(-z))
            prod = yf_train-h
            temp_bias = temp_bias + (lr_rate/10)*(np.sum(prod)/xf_train.shape[0])
            for k in range(len(temp_theta)):
                x_temp = xf_train[:,k:k+1].reshape(xf_train.shape[0])
                prod_temp = prod*x_temp
                temp_theta[k] = temp_theta[k] + (lr_rate/10)*(np.sum(prod_temp)/xf_train.shape[0])
                
            
        z_valid = xf_valid.dot(temp_theta) + temp_bias
        y_pred = 1/(1+np.exp(-z_valid))
        for k in range(len(y_pred)):
            if(y_pred[k]>0.5):
                y_pred[k] = 1
            else:
                y_pred[k] = 0
                
        valid_acc = sum(y_pred==yf_valid)/len(y_pred)
        fold_acc.append(valid_acc)
        
    avg_acc = np.sum(fold_acc)/len(fold_acc)
    accuracies.append(avg_acc)

best_lr_rate = 0.0
best_acc = 0
for i in range(len(accuracies)):
    if(best_acc<=accuracies[i]):
        best_acc = accuracies[i]
        best_lr_rate = lr_rate_range[i]/10    
    
best_lr_rate = 1.0
    
y_pred = []
loss_hist = []
train_acc = []

for itr in range(num_epochs):
    z = x_train.dot(theta) + bias
    h = 1/(1+np.exp(-z))
    y_pred_train = []
    loss_temp = 0.0
    for i in range(len(h)):
        if(h[i]>0.5):
            y_pred_train.append(1)
        else:
            y_pred_train.append(0)
            
        loss_temp += -(y_train[i]*np.log(h[i]) + (1-y_train[i])*np.log(1-h[i]))
    
    train_accuracy = np.sum(y_pred_train==y_train)/len(y_train)            
    loss_temp /= x_train.shape[0]
    loss_hist.append(loss_temp)
    prod = y_train-h
    bias = bias + best_lr_rate*(np.sum(prod)/len(prod))
    
    for k in range(len(theta)):
        x_temp = x_train[:,k:k+1].reshape(x_train.shape[0])
        prod_temp = prod*x_temp
        theta[k] = theta[k] + best_lr_rate*(np.sum(prod_temp)/x_train.shape[0])
    
    print("train accuracy:", train_accuracy)
    train_acc.append(train_accuracy)

#plot of loss vs iteration
index = range(1,len(loss_hist)+1)
plt.plot(index,loss_hist)
plt.xlabel("Number of Iterarion")
plt.ylabel("loss")
plt.show()
     
z_pred = x_test.dot(theta) + bias
h_pred = 1/(1+np.exp(-z_pred))
for i in range(len(h_pred)):
    if(h_pred[i]>0.5):
        y_pred.append(1)
    else:
        y_pred.append(0)

test_acc = np.sum(y_pred==y_test)/len(y_test)

#plot of train accuracy vs iteration
plt.plot(index,train_acc)
plt.xlabel("Number of Iterarion")
plt.ylabel("train_accuracy")
plt.show()




            
            
            
        
    

        
        
        
        
        
        
        
        
        
