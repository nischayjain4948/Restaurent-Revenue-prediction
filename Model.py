import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


### probleme statement === This is a dataset of Restaurent and we have to predict the revenu of the restaurent!

train_data = pd.read_csv('train.csv',index_col=0)

test_data = pd.read_csv('test.csv',index_col=0)
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')




test_data = pd.concat([test_data,pd.get_dummies(test_data['Type'])],axis=1)
train_data = pd.concat([train_data,pd.get_dummies(train_data['Type'])],axis=1)
test_data.drop('Type',axis=1,inplace=True)
train_data.drop('Type',axis=1,inplace=True)


test_data.drop('MB',axis=1,inplace=True)

test_data['New City Group'] = pd.get_dummies(test_data['City Group'],drop_first=True)
train_data['New City Group'] = pd.get_dummies(train_data['City Group'],drop_first=True)
test_data.drop('City Group',axis=1,inplace=True)
train_data.drop('City Group',axis=1,inplace=True)


train_data.head(10)

test_data = pd.concat([test_data,pd.get_dummies(test_data['City'])],axis=1)
train_data = pd.concat([train_data,pd.get_dummies(train_data['City'])],axis=1)
test_data.drop('City',axis=1,inplace=True)
train_data.drop('City',axis=1,inplace=True)

for i in test_data.columns:
    if( (i not in train_data.columns) and (i != 'revenue')):
        test_data.drop(i,axis=1,inplace=True)


len(train_data.columns)

len(test_data.columns)

train_data['Open Date'][0].split('/')

train_data['Date'] = train_data['Open Date'].apply(lambda x: int(x.split('/')[0]))

train_data['Month'] = train_data['Open Date'].apply(lambda x: int(x.split('/')[1]))

train_data['Year'] = train_data['Open Date'].apply(lambda x: int(x.split('/')[2]))

train_data.head()

test_data['Date'] = test_data['Open Date'].apply(lambda x: int(x.split('/')[0]))


test_data['Month'] = test_data['Open Date'].apply(lambda x: int(x.split('/')[1]))


test_data['Year'] = test_data['Open Date'].apply(lambda x: int(x.split('/')[2]))



train_data.drop('Open Date',axis=1,inplace=True)
test_data.drop('Open Date',axis=1,inplace=True)


import seaborn as sns
print(train_data['revenue'].describe())
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    train_data['revenue'], norm_hist=False, kde=True
).set(xlabel='revenue', ylabel='P(revenue)');

features = train_data.drop('revenue',axis=1)

labels = train_data['revenue']
labels

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(features,labels)

model.coef_

pd.DataFrame(model.coef_,columns=['Coef'],index= features.columns).sort_values(by='Coef',ascending=False)[0:20]

print('Model Score: ' + str(round(model.score(features,labels)*100,2)))

train_data['revenue'].hist()

train_data[train_data['revenue']>10000000]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse

from sklearn.tree import DecisionTreeRegressor
X = train_data.drop('revenue',axis=1)
y = train_data['revenue']
tree = DecisionTreeRegressor().fit(X, y)

import numpy as np
d = {'Importance': np.round(tree.feature_importances_, 3) , 'Features': X.columns}
feature_imp = pd.DataFrame(data=d)
feature_imp

feature_imp_order = feature_imp.sort_values('Importance', ascending=False).reset_index(drop=True)
feature_imp_order.Features[0:15]

X_main = train_data[['P29','Year','P27','İzmir','İstanbul','Date']]
y_main = train_data['revenue']
scaler_train = StandardScaler()
scaler_train.fit(X_main)
X_main_scaled = scaler_train.transform(X_main)

# Splitting the data into training and testing data
linear_regr = LinearRegression()
linear_regr.fit(X_main_scaled, y)
y_pred = linear_regr.predict(X_main_scaled)
accuracy = linear_regr.score(X_main_scaled,y)
print("Train Accuracy {}%".format(int(round(accuracy *100))))
print("Training RMSE Linear regression ",mse(y, y_pred)**0.5)

from sklearn.metrics import mean_absolute_error as mae
MAE = mae(y, y_pred)
print("MAE: {0}".format(MAE))

final_model = linear_regr    # saving final Model
pd.to_pickle(final_model, 'RestaurantRevenuePrediction')

model = pd.read_pickle('RestaurantRevenuePrediction')

P29 = eval(input(''))
Year = eval(input(''))
P27 = eval(input(''))
İzmir =  eval(input(''))
İstanbul = eval(input(''))
Date = eval(input(''))
query = pd.DataFrame({
    'P29':[P29],
    'Year':[Year],
    'P27':[P27],
    'İzmir': [İzmir],
    'İstanbul': [İstanbul],
    'Date': [Date]})

query


print('Predicted Revenue : ' + str(round(model.predict(query)[0],2)))

