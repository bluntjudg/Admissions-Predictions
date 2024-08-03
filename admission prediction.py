import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate random data
np.random.seed(0)
n = 1000  # Number of samples

X1 = np.random.uniform(0, 100, n)  # Entrance exam scores
X2 = np.random.uniform(0, 100, n)  # Percentage

# Simulate admission based on some rule (e.g., a linear combination of X1 and X2)
admission_prob = 1 / (1 + np.exp(-(0.1*X1 + 0.2*X2 - 10)))  
y = np.random.binomial(1, admission_prob)


data = pd.DataFrame({'Entrance Exam Score': X1, 'Percentage': X2, 'Admission': y})

X = data[['Entrance Exam Score', 'Percentage']]
y = data['Admission']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(data)
print('Logistic Regression')
print('actual out:',list(y_test))
print('predict out:',y_pred)
print('accuracy:',model.score(X_train,y_train))




from sklearn import svm
model=svm.SVC()
model.fit(X_train,y_train)
pred=model.predict(X_test)
print('svm')
print('actual out:',list(y_test))
print('predict out:',y_pred)
print('accuracy:',model.score(X_train,y_train))




from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('DT')
print('actual out:', list(y_test))
print('predict out:',y_pred)
print('accuracy:', model.score(X_train, y_train))


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion ='entropy', random_state=0) # only 10 trees will b ok as dataset is only 400+ records. Default is 100.    
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print('RF')
print('actual out:', list(y_test))
print('predict out:', y_pred)
print('accuracy:', model.score(X_train, y_train))



from sklearn import neighbors
model=neighbors.KNeighborsClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print('KNN')
print("kNN\nActual output : ",list(y_test))
print("Predicted output:",y_pred)
print("Accuracy:",model.score(X_train,y_train))

