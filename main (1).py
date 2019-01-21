import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

inputdata = pd.read_csv('input2012and13.csv', header = 0)
inputdata2 = inputdata.iloc[0:17208,1:7] # column 0 was date column
print(inputdata2.head()) 
inputdata2 = inputdata2.values
print(type(inputdata2))

targetdata = inputdata.iloc[0:17208,7]
targetdata = targetdata.values

X_test1 = pd.read_csv('input2014.csv', header = 0)
X_test = X_test1.iloc[0:8592, 1:7]
X_test = X_test.values
y_test = X_test1.iloc[0:8592, 7]
y_test = y_test.values

X_train = inputdata2
y_train = targetdata

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(inputdata2, targetdata)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)      
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPRegressor
mlp =  MLPRegressor(hidden_layer_sizes=(50), activation = 'logistic', solver = 'lbfgs', max_iter = 10000)
mlp.fit(X_train, y_train)

# x = day of the year
x = 61
start = 24 * x
end = 24 * (x + 1)
predictionsday = mlp.predict(X_test[start:end])
print(predictionsday)

dateselect = X_test1.iloc[start:end,0]
dateselect = dateselect.values
plt.plot(predictionsday, color = 'blue')
plt.plot(y_test[(24*x):end], color = 'red')
plt.ylabel('test load 2014 for day')
plt.show()
print('MAPE')
mape1 = np.mean(np.absolute((y_test[start : end]- predictionsday) / y_test[start : end])) * 100
print (mape1)
x0 = np.abs((y_test[start : end] - predictionsday) / y_test[start : end])*100         
plt.plot(x0)
plt.ylabel('MAPE error')
plt.show()

# weekly----- y = week of the year
y = 10
start = (y*24*7)+1
end = (y+1)*24*7
        
predictionsweek = mlp.predict(X_test[start : end])
print(predictionsweek)

plt.plot(predictionsweek, color = 'blue')
plt.plot(y_test[start : end], color = 'red')
plt.ylabel('test load 2014 for week')
plt.show()
print('MAPE')
mape1 = np.mean(np.absolute((y_test[start : end]- predictionsweek) / y_test[start : end])) * 100
print (mape1)
x0 = np.abs((y_test[start : end] - predictionsweek) / y_test[start : end])*100         
plt.plot(x0)
plt.ylabel('MAPE error')
plt.show()

predictions = mlp.predict(X_test)
predictions2 = mlp.predict(X_train)
#mlp.score(predictions, y_test)
plt.plot(y_test, color = 'red')
plt.plot(predictions, color = 'blue')
plt.ylabel('Test Load 2014')
plt.show()
plt.plot(y_train, color='red')
plt.plot(predictions2, color='green')
plt.ylabel('Training Load 2012 and 2013')
plt.show()
#metrics to evaluate performance 
print('MAPE')
mape1 = np.mean(np.absolute((y_test - predictions) / y_test)) * 100
print (mape1)

print ('RMSE of testing data')
#error1 = ((predictions-y_test)**2)
#print(error1.sum()/len(y_test))         
from sklearn.metrics import mean_squared_error
from math import sqrt

print(sqrt(mean_squared_error(y_test, predictions)))

x0 = np.abs((y_test - predictions) / y_test)*100         
plt.plot(x0)
plt.ylabel('MAPE error')
plt.show()                 
data_to_plot = [x0]
import matplotlib.pyplot as plt1    

fig = plt1.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(data_to_plot)
fig.savefig('fig1.png', bbox_inches='tight')

