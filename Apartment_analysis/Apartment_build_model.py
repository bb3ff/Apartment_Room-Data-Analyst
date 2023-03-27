# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:14:33 2023

@author: ASUS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Apartment_cleaned_data.csv')

# feature selection
df.dtypes
df_model=df[['building_type', 'building_materials','room_floor','house_plan', 'have_living_room', 'have_kitchen_room', 
             'have_dining_room','building_age_int', 'management_fee_int','building_floor_int','house_area_int','room_floor_int','rent_fee_int']]

df_dum=pd.get_dummies(df_model)


df_dum[df_dum.columns[:-1]].values
X=df_dum.drop(['rent_fee_int'],axis=1)
y=df_dum['rent_fee_int']
# split data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

#build model
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
lm= LinearRegression()
lm.fit(X_train,y_train)
np.mean(cross_val_score(lm,X_train,y_train,scoring='neg_mean_absolute_error', cv=3))

lm_l=Lasso(alpha=0.15)
lm_l.fit(X_train,y_train)

alpha=[]
error=[]
for i in range(1,100):
    alpha.append(i/100)
    lm_la=Lasso(alpha=i/100)
    error.append(np.mean(cross_val_score(lm_la, X_train,y_train,scoring='neg_mean_absolute_error', cv=3)))

plt.plot(alpha,error)
plt.show()

err=tuple(zip(alpha,error))
df_err=pd.DataFrame(err,columns=['alpha','error'])
df_err[df_err.error==max(df_err.error)]

from sklearn.ensemble import RandomForestRegressor

rf_r=RandomForestRegressor()
np.mean(cross_val_score(rf_r,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))
rf_r.fit(X_train, y_train)

import statsmodels.api as sm
X_sm=sm.add_constant(X)
model=sm.OLS(y,X_sm)
model.fit().summary()

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)


#Neural network 
import tensorflow as tf

def plot_history(history):
    fig, (ax1, ax2)= plt.subplots(nrows=2, ncols=1)
    
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Square Error')
    ax1.grid(True)

     
    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.grid(True)
    plt.show()

#temp_normalizer= tf.keras.layers.Normalization(input_shape=(11,1), axis=-1)
#temp_normalizer.adapt(X_train.reshape(-1))
def nn_model_train(X_train,y_train,num_nodes,lr,batch_size,epochs):
    nn_model=tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dense(1)
        
        
        ])
    
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mean_squared_error')
    
    history=nn_model.fit(X_train, y_train, batch_size=batch_size,verbose=0,validation_split=0.2, epochs=epochs)
    
    return nn_model,history

least_val_loss=float('inf')
least_loss_model=None
epochs=100
for num_nodes in [16, 32, 64, 128]:
    for lr in [0.01, 0.001, 0.005]:
        for batch_size in [16,32,48,64,128]:
            print(f"num_nodes {num_nodes}, lr {lr}, batch_size {batch_size}")
            nn_model,history=nn_model_train(X_train,y_train,num_nodes,lr,batch_size,epochs)
            
            plot_loss(history)
            val_loss=nn_model.evaluate(X_test,y_test)
            if val_loss<least_val_loss:
                least_loss_model=val_loss
                least_loss_model=nn_model
                
# =============================================================================
# plt.scatter(X_train, y_train)
# x=tf.linspace(0,80,100)
# plt.plot(x,least_loss_model.predict(np.array(x).reshape(-1,1)), label='fit', linewidth=3)
# plt.show()
# =============================================================================

# predict
t_pred_lm=lm.predict(X_test)
t_pred_lasso=lm_l.predict(X_test)
t_pred_rf_r=rf_r.predict(X_test)
t_pred_nn_model=least_loss_model.predict(X_test)
print(t_pred_nn_model)

#plot 
plt.scatter(X_test['house_area_int'],y_test, label='Real Data')
plt.scatter(X_test['house_area_int'], t_pred_lm, label='Lin Reg Pred')
plt.scatter(X_test['house_area_int'], t_pred_nn_model, label='NN model Pred' )
plt.scatter(X_test['house_area_int'], t_pred_rf_r, label='Random Forest Pred' )
plt.legend()
# =============================================================================
# lims=[0,500]
# plt.xlim(lims)
# plt.ylim(lims)
# plt.plot(lims, lims, c='red')
# =============================================================================

lm.score(X_test,y_test)
lm_l.score(X_test,y_test)
rf_r.score(X_test,y_test)
dt.score(X_test,y_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error
mean_absolute_error(y_test,t_pred_lm)
mean_absolute_error(y_test,t_pred_lasso)
mean_absolute_error(y_test,t_pred_rf_r)
mean_absolute_error(y_test,t_pred_nn_model)

mean_squared_error(y_test,t_pred_lm)
mean_squared_error(y_test,t_pred_nn_model)
mean_squared_error(y_test, t_pred_rf_r)