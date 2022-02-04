#!/usr/bin/env python
# coding: utf-8

# In[853]:


# import required library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import warnings
warnings.filterwarnings('ignore')


# In[854]:


# import dataset
df=pd.read_csv('Flightprice.csv')
df


# ### column name
#     1.name => airlines name
#     2.deprt => depature time
#     3.arrvt => arrival time
#     4.sorct => source city
#     5.dest => destination city
#     6.stops => No.of stops
#     7.duration => total duration traveling time
#     8.price => traveling charges of airlines 

# In[855]:


# dataset information
df.info()


# In[856]:


# checking null values
df.isnull().sum()


# In[857]:


# checking duplicates values
df.duplicated().sum()


# In[858]:


# check data types
df.dtypes


# - here we see that price, deprt,arrvt,durat are in objective so cinvert into int and datetime format

# ### data preprocessing

# In[859]:


# remove ',' from price column and change data type to int
df.replace(',','',regex=True,inplace=True)


# In[860]:


df['price']=df['price'].astype('int')


# In[861]:


# slip arrival time column to get specific time
new = df["arrvt"].str.split("\n", n = 1, expand = True)


# In[862]:


df["arrvtime"]= new[0]
df.drop(['arrvt'],axis=1,inplace=True)


# In[863]:


# remove unnecessary data
df.drop('Unnamed: 0',axis=1,inplace=True)


# In[864]:


# change data type of arrival time into hours and min
df["arrvtime_h"] = pd.to_datetime(df['arrvtime']).dt.hour
df["arrvtime_m"] = pd.to_datetime(df['arrvtime']).dt.minute


# In[865]:


# change data type of departure time into hours and min
df['deprt_h']=pd.to_datetime(df['deprt']).dt.hour
df['deprt_m']=pd.to_datetime(df['deprt']).dt.minute


# In[866]:


# drop axis after conversion
df.drop(['deprt','arrvtime'],axis=1,inplace=True)


# In[867]:


# value count of No of stops
df['stops'].value_counts()


# In[868]:


# assign the value to no of stop columns
df['stops']=df['stops'].replace({'Non Stop':0,'1 Stop':1,'2 Stop(s)':2,
                                '3 Stop(s)':3,'4 Stop(s)':4})


# In[869]:


# convert total travelling time into hours
df['durat_h']=(pd.to_timedelta(df['durat']).dt.seconds/3600).astype('float32')


# In[870]:


# drop column after get info
df.drop('durat',axis=1,inplace=True)


# In[871]:


df['durat_h']=df['durat_h'].round(decimals =2)


# In[872]:


df


# In[873]:


# agin check data types
df.dtypes


# ### EDA

# In[874]:


df['name'].unique()


# In[875]:


df['sorct'].unique()


# In[876]:


df['dest'].unique()


# In[877]:


plt.figure(figsize=(10,7))
ax=sns.countplot(x='name',data=df,palette='Spectral')
plt.title('Toatal Airlines',fontsize=15)
plt.xticks(rotation=90)
plt.show()


# In[878]:


plt.figure(figsize=(10,7))
ax=sns.barplot(x='name',y='price',data=df,palette='Spectral')
plt.xticks(rotation=90)
plt.title("Airlines Vs Price",fontsize=15)
plt.show()


# In[879]:


plt.figure(figsize=(10,7))
ax=sns.countplot(x='stops',data=df,palette='Spectral')
plt.title("No. of stops",fontsize=15)
plt.xticks(rotation=90)
plt.show()


# In[880]:


plt.figure(figsize=(10,7))
ax=sns.barplot(x='stops',y='price',data=df,palette='Spectral')
plt.title("Stops Vs Price",fontsize=20)
plt.xticks(rotation=90)
plt.show()


# In[881]:


plt.figure(figsize=(10,7))
ax=sns.countplot(x='sorct',data=df,palette='Spectral')
plt.title("No. of sources",fontsize=15)
plt.xticks(rotation=90)
plt.show()


# In[882]:


plt.figure(figsize=(10,7))
ax=sns.countplot(x='dest',data=df,palette='Spectral')
plt.title("No. of sources",fontsize=15)
plt.xticks(rotation=90)
plt.show()


# In[883]:


plt.figure(figsize=(10,7))
ax=sns.barplot(x='sorct',y='price',data=df,palette='Spectral')
plt.title("Source Vs Price",fontsize=20)
plt.xticks(rotation=90)
plt.show()


# In[884]:


plt.figure(figsize=(10,7))
ax=sns.barplot(x='dest',y='price',data=df,palette='Spectral')
plt.title("Destination Vs Price",fontsize=20)
plt.xticks(rotation=90)
plt.show()


# In[885]:


plt.figure(figsize=(10,7))
ax=sns.barplot(x='name',y='price',hue='stops',data=df,palette='Spectral')
plt.xticks(rotation=90)
plt.title('Price variance with no of stops')
plt.xlabel('Airlines names',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.show()


# In[886]:


plt.figure(figsize=(15,7))
sns.countplot(x ="sorct",hue ="name",data = df)
plt.title('Different airlines from sources city')
plt.xlabel('Source',fontsize=15)
plt.show()


# In[887]:


plt.figure(figsize=(15,7))
sns.countplot(x = "dest",hue ="name",data = df)
plt.title('Different airlines from the destination city')
plt.xlabel('Destination',fontsize=20)
plt.show()


# In[888]:


df.columns


# In[889]:


plt.figure(figsize=(15,7))
ax=sns.barplot(x='name',y='price',data=df.loc[df['durat_h']>=15],hue='stops',palette='Spectral')
plt.xticks(rotation=90)
plt.title('Which airlines take more hours by no of stops',fontsize=15)
plt.xlabel('Airlines Name',fontsize=15)
plt.ylabel('Prices',fontsize=15)
plt.show()


# In[890]:


min_max_price = pd.DataFrame(df.groupby(['name','sorct','dest',]).price.agg([min,max]).sort_values(by='max',ascending=False))


# In[891]:


min_max_price


# In[892]:


# box plot for outliers
for i in df.columns:
    if df[i].dtypes != 'O':
        plt.plot()
        df[i].plot(kind='box',)
        plt.legend([i])
        plt.show()


# In[893]:


# distribution plot for numerical variables
for i in df.columns:
    if df[i].dtypes != 'O':
        plt.plot()
        sns.distplot(df[i])
        plt.legend([i])
        plt.show()


# In[894]:


df.describe()


# In[896]:


#lets convert categorical data into numeric values, using OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
for i in df.columns:
    if df[i].dtypes == "object" :
        df[i] = enc.fit_transform(df[i].values.reshape(-1,1))


# In[897]:


df


# In[898]:


df.corr()['price']


# In[899]:


#Lets plot heatmap to check correlation among differnt features and label
df_corr = df.corr()
plt.figure(figsize = (20,10))
sns.heatmap(df_corr,vmin=-1,vmax=1,annot=True,square=True,center=0,fmt='.2g',linewidths=0.1)
plt.tight_layout()


# - from the correlation map we say that no.of stops,duration hours are most correlated with target price
# - other are less correlation with it

# In[900]:


#I will shuffle our data for getting good result while evaluating
df = df.sample(frac = 1)
df.reset_index(inplace = True)
df.drop(columns = 'index', inplace = True)


# In[901]:


#apply zscore to remove outliers
from scipy import stats
from scipy.stats import zscore
z_score = zscore(df[['price']])
abs_z_score = np.abs(z_score)
filtering_entry = (abs_z_score < 3).all(axis = 1)
df = df[filtering_entry]
df.shape


# In[902]:


df


# In[903]:


x = df.drop(columns = 'price')
y = df['price']


# In[904]:


x


# In[905]:


y


# In[906]:


x.skew()


# In[907]:


#Lets treat the skewness
for index in x.skew().index:
    if x.skew().loc[index]>0.5:
        x[index]=np.log1p(x[index])
        if x.skew().loc[index]<-0.5:
            x[index]=np.square(x[index])


# In[908]:


x.skew()


# In[914]:


#lets apply standard scaler to numerical features to bring them to common scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(x)


# In[911]:


from sklearn.linear_model import LinearRegression,SGDRegressor,Lasso,Ridge,BayesianRidge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor,XGBRFRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
lr=LinearRegression()


# ### select best random state

# In[916]:


for i in range(1,500):
    x_train,x_test,y_train,y_test=train_test_split(X,np.log(y),test_size=0.2,random_state=i)
    lr.fit(x_train,y_train)
    pred_tr=lr.predict(x_train)
    pred_te=lr.predict(x_test)
    if round(r2_score(y_train,pred_tr)*100,1)==round(r2_score(y_test,pred_te)*100,1):
        print('Random state',i)
        print('Train score',r2_score(y_train,pred_tr)*100)
        print('Test score',r2_score(y_test,pred_te)*100)


# In[917]:


x_train,x_test,y_train,y_test=train_test_split(X,np.log(y),test_size=0.20,random_state=229)
lr.fit(x_train,y_train)
pred_lr=lr.predict(x_test)
print('train score',lr.score(x_train,y_train)*100)
print('R2 score',r2_score(y_test,pred_lr)*100)
print('RMSE',np.sqrt(mean_squared_error(y_test,pred_lr)))
print('Absolute error',mean_absolute_error(y_test,pred_lr))


# ### parameter selection

# In[918]:


parameters={'kernel':['poly','rbf','linear'],'C':[0.1,0.01,1,10,0.001]}
clf = GridSearchCV(SVR(), parameters, cv=5,scoring="r2")
clf.fit(X,np.log(y))
clf.best_params_


# In[919]:


neighbors={"n_neighbors":range(1,30)}
clf = GridSearchCV(KNeighborsRegressor(),neighbors, cv=5,scoring="r2")
clf.fit(X,np.log(y))
clf.best_params_


# In[920]:


pr={'alpha':[0.1,0.01,0.001],'fit_intercept':[True,False],'normalize':[True,False],'precompute':['auto',True,False]}
clf = GridSearchCV(Lasso(), pr,scoring="r2",cv=5)
clf.fit(X,np.log(y))
print(clf.best_params_)


# ### Model building

# In[922]:


dtc=DecisionTreeRegressor(random_state=165)
svr=SVR(kernel="rbf",C=10)
kn=KNeighborsRegressor(n_neighbors=1)
l1=Lasso(alpha=0.001,fit_intercept=True,normalize=False,precompute=True)
l2=Ridge(alpha=0.1,solver="lsqr")
sgd=SGDRegressor()
xgb=XGBRegressor()


# In[925]:


dtc=DecisionTreeRegressor()
dtc.fit(x_train,y_train)
print(dtc,'\n',dtc.score(x_train,y_train)*100)
pred_dtc=dtc.predict(x_test)
crs_score=cross_val_score(dtc,X,np.log(y),cv=5,scoring='r2')
print('cross value score',crs_score.mean())
print('R2_score :',r2_score(y_test,pred_dtc)*100)
print('error1:\n:',mean_absolute_error(y_test,pred_dtc))
print('RSME:\n:',np.sqrt(mean_squared_error(y_test,pred_dtc)))


# In[926]:


svr=SVR(kernel='linear',C=10)
svr.fit(x_train,y_train)
print(svr,'\n',svr.score(x_train,y_train)*100)
pred_svr=svr.predict(x_test)
crs_score=cross_val_score(svr,X,np.log(y),cv=5,scoring='r2')
print('cross value score',crs_score.mean())
print('R2_score :',r2_score(y_test,pred_svr)*100)
print('error1:\n:',mean_absolute_error(y_test,pred_svr))
print('RSME:\n:',np.sqrt(mean_squared_error(y_test,pred_svr)))


# In[927]:


kn=KNeighborsRegressor(n_neighbors=3)
kn.fit(x_train,y_train)
print(kn,'\n',kn.score(x_train,y_train)*100)
pred_kn=kn.predict(x_test)
crs_score=cross_val_score(kn,X,np.log(y),cv=5,scoring='r2')
print('cross value score',crs_score.mean())
print('R2_score :',r2_score(y_test,pred_kn)*100)
print('error1:\n:',mean_absolute_error(y_test,pred_kn))
print('RSME:\n:',np.sqrt(mean_squared_error(y_test,pred_kn)))


# In[928]:


l1.fit(x_train,y_train)
print(l1,'\n',l1.score(x_train,y_train)*100)
pred_l1=l1.predict(x_test)
crs_score=cross_val_score(l1,X,np.log(y),cv=5,scoring='r2')
print('cross value score',crs_score.mean())
print('R2_score :',r2_score(y_test,pred_l1)*100)
print('error1:\n:',mean_absolute_error(y_test,pred_l1))
print('RSME:\n:',np.sqrt(mean_squared_error(y_test,pred_l1)))


# In[929]:


l2.fit(x_train,y_train)
print(l2,'\n',l2.score(x_train,y_train)*100)
pred_l2=l2.predict(x_test)
crs_score=cross_val_score(l2,X,np.log(y),cv=5,scoring='r2')
print('cross value score',crs_score.mean())
print('R2_score :',r2_score(y_test,pred_l2)*100)
print('error1:\n:',mean_absolute_error(y_test,pred_l2))
print('RSME:\n:',np.sqrt(mean_squared_error(y_test,pred_l2)))


# In[930]:


sgd.fit(x_train,y_train)
print(sgd,'\n',sgd.score(x_train,y_train)*100)
pred_sgd=sgd.predict(x_test)
crs_score=cross_val_score(sgd,X,np.log(y),cv=5,scoring='r2')
print('cross value score',crs_score.mean())
print('R2_score :',r2_score(y_test,pred_sgd)*100)
print('error1:\n:',mean_absolute_error(y_test,pred_sgd))
print('RSME:\n:',np.sqrt(mean_squared_error(y_test,pred_sgd)))


# In[931]:


xgb.fit(x_train,y_train)
print(xgb,'\n',xgb.score(x_train,y_train)*100)
pred_xgb=dtc.predict(x_test)
crs_score=cross_val_score(xgb,X,np.log(y),cv=5,scoring='r2')
print('cross value score',crs_score.mean())
print('R2_score :',r2_score(y_test,pred_xgb)*100)
print('error1:\n:',mean_absolute_error(y_test,pred_xgb))
print('RSME:\n:',np.sqrt(mean_squared_error(y_test,pred_xgb)))


# In[932]:


# use ensemble methods
ensemble=[RandomForestRegressor(),AdaBoostRegressor(),GradientBoostingRegressor(),BaggingRegressor()]


# In[933]:


for i in ensemble:
    i.fit(x_train,y_train)
    print('\n \nscore: ',i,':\n',i.score(x_train,y_train)*100)
    pred=i.predict(x_test)
    print(' R2 score:',r2_score(y_test,pred)*100)
    print('error1:\n',mean_absolute_error(y_test,pred))
    print('RSME:\n',np.sqrt(mean_squared_error(y_test,pred)))


# ### Select best model for hypertunning

# In[944]:


#lets selects different parameters for tuning
grid_params = {
                'n_estimators': [50,100,500,700],
                'criterion' : ["mse", "mae"],
                'max_features' : ["auto", "sqrt", "log2"],
                }


# In[945]:


#train the model with given parameters using GridSearchCV
clf=  GridSearchCV(RandomForestRegressor(), grid_params,verbose=1,refit=True,n_jobs=-1, cv = 5)
clf.fit(x_train,y_train)


# In[946]:


clf.best_params_  


# In[947]:


clf.best_score_


# In[950]:


rf=RandomForestRegressor(criterion='mse',max_features='sqrt',n_estimators=700)
rf.fit(x_train,y_train)
print(rf,'\n',rf.score(x_train,y_train)*100)
pred_rf=rf.predict(x_test)
crs_score=cross_val_score(rf,X,np.log(y),cv=5,scoring='r2')
print('cross value score',crs_score.mean())
print('R2_score :',r2_score(y_test,pred_rf)*100)
print('error1:\n:',mean_absolute_error(y_test,pred_rf))
print('RSME:\n:',np.sqrt(mean_squared_error(y_test,pred_rf)))


# In[956]:


data = pd.DataFrame({'Y Test':y_test , 'Pred':pred_rf},columns=['Y Test','Pred'])
sns.lmplot(x='Y Test',y='Pred',data=data,palette='rainbow')
plt.title('Actual Vs Prediction graph')
plt.show()


# In[953]:


predicted_prices = np.exp(pred_rf)
predicted_prices


# In[955]:


# saving model
import pickle
pickle.dump(rf,open('Flight_price_FR.pkl','wb'))
rf_model=pickle.load(open('Flight_price_FR.pkl','rb'))


# In[ ]:




