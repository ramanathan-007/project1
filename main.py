import numpy
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pandas
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import bar_chart_race
import plotly
#svm

import sklearn.metrics
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

dataset = [5,7,8,7,2,17,2,9,4,11,12,9,6]

x=numpy.mean(dataset)
print("Mean:",x)

y=numpy.median(dataset)
print("Median:",y)

z=stats.mode(dataset)
print("Mode:",z)

s=numpy.std(dataset)
print("Standard Deviation:",s)

v=numpy.var(dataset)
print("Variance:",v)

p = numpy.percentile(dataset, 75)
print("Percentile:",p)

plt.hist(dataset,100)
plt.show()

m = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(dataset,m)
plt.show()

#Linear Regression

slope, intercept, r, p, std_err = stats.linregress(dataset,m)

def linearreg(dataset):
    return slope * dataset +intercept

model1= list(map(linearreg, dataset))

plt.scatter(dataset,m)
plt.plot(dataset,model1)
plt.show()


#Polynomial Regression

model2= numpy.poly1d(numpy.polyfit(dataset,m,3))
line1=numpy.linspace(1,25,100)

plt.scatter(dataset,m)
plt.plot(line1, model2(line1))
plt.show()

e = model2(20)
print("polynomiqal regression:",e)


#multiple regression

df=pandas.read_csv("C:\\Users\\raman\\Desktop\\data.csv")

xa = df[['goals%','freethrows%']]
xb = df['points']
print(df.head())
print(df.describe())

regr = linear_model.LinearRegression()
regr.fit(xa,xb)

predictedpoints =regr.predict([[0.5,0.7]])

print("predicted points:",predictedpoints)
print("reg coeff:", regr.coef_)


#scaling

scale= StandardScaler()

xm = df[['Height','weight']]

scaleda = scale.fit_transform(xm)

print("Scale",scaleda)

#barchart
plt.bar(xb,100)
plt.show()

#support vector Machine.

cancer_data = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.4, random_state=209)

cls=SVC(kernel="linear")

cls.fit(x_train,y_train)

pred =cls.predict(x_test)

print("accuracy:", sklearn.metrics.accuracy_score(y_test, y_pred=pred))

print("precision:",sklearn.metrics.precision_score(y_test, y_pred=pred))

print("recall:", sklearn.metrics.recall_score(y_test, y_pred=pred))

print(sklearn.metrics.classification_report(y_test, y_pred=pred))



digits = load_digits()

clf1= SVC(gamma=0.00000000001, C=100)
x1,y1= digits.data[:10], digits.target[:10]
clf1.fit(x1,y1)

print(clf1.predict(digits.data[:10]))
plt.imshow(digits.images[7],interpolation='nearest')
plt.show()





