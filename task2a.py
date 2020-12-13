import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

#reading in the files
life = pd.read_csv('life.csv',encoding = 'ISO-8859-1',index_col="Country Code")
world = pd.read_csv('world.csv',encoding = 'ISO-8859-1',index_col="Country Code")

#merge them
result = [life, world]
result = pd.concat(result, axis=1, join='inner')
result = result.drop(["Country Name","Time","Country","Year"], axis = 1)

#seperate the merged table into classlabel and feactures groups
data = [result for result in result.keys()]
classlabel = result[data[0]]
data = result[data[1:]]
X_train, X_test, y_train, y_test = train_test_split(data,classlabel, train_size=(2/3), test_size=(1/3), random_state=100)

# compute X_train(replace .. with median)
X_train = X_train.replace('..',np.nan)
imp = SimpleImputer(missing_values = np.nan, strategy="median")  
idf=pd.DataFrame(imp.fit_transform(X_train))
idf.columns=X_train.columns
idf.index=X_train.index
X_train = idf

#impute X_test(replace .. with its medina)
X_test = X_test.replace('..',np.nan)
temp=pd.DataFrame(imp.fit_transform(X_test))
temp.columns=X_test.columns
temp.index=X_test.index
X_test = temp

#writing output
with open('task2a.csv', 'w+', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['feature',"median","mean","variance"])
    for items in X_train:
        writer.writerow([items,X_train[items].median(),X_train[items].mean(),X_train[items].var()])

#normalization
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#tree-depth 3
dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, y_train)
tree_pred=dt.predict(X_test)
print("Accuracy of decision tree: "+ str(round(accuracy_score(y_test, tree_pred)*100,3))+"%")

#knn-5
kn5= neighbors.KNeighborsClassifier(n_neighbors=5)
kn5.fit(X_train, y_train)
kn5_pred=kn5.predict(X_test)
print("Accuracy of k-nn (k=5): "+ str(round(accuracy_score(y_test, kn5_pred)*100,3))+"%")

#knn-10
kn10 = neighbors.KNeighborsClassifier(n_neighbors=10)
kn10.fit(X_train, y_train)
kn10_pred=kn10.predict(X_test)
print("Accuracy of k-nn (k=10): "+ str(round(accuracy_score(y_test, kn10_pred)*100,3))+"%")

