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
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA

#read in the files and join them together
life = pd.read_csv('life.csv',encoding = 'ISO-8859-1',index_col="Country Code")
world = pd.read_csv('world.csv',encoding = 'ISO-8859-1',index_col="Country Code")
result = [life, world]
result = pd.concat(result, axis=1, join='inner')

#drop whatever we dont want
result = result.drop(["Country Name","Time","Country","Year"], axis = 1)
result = result.replace('..',np.nan)

#copying the entire table for feature engineeringf, pca, and first 4 feacture
top4 = result
pcadata = result
data = [str(result) for result in result.keys()]
datakeys = data[1:]



############### feacture engineering ################

#generate f1xf2 features
for i in range(0,len(datakeys)-1):
    for g in range(i + 1, len(datakeys)):

        temp = ''
        temp = temp + datakeys[i] + " x " + datakeys[g]        
        result[temp]  = (result[datakeys[i]]).astype(float) * (result[datakeys[g]]).astype(float)

#imputing data, replacing the null with median
data = [str(result) for result in result.keys()]
classlabel = result[data[0]]
data = result[data[1:]]
imp = SimpleImputer(missing_values = np.nan, strategy="median")  
idf=pd.DataFrame(imp.fit_transform(data))
idf.columns=data.columns
idf.index=data.index
data = idf

#generate clustering feacture
kmean = KMeans(n_clusters=3,max_iter = 1000).fit(data)
cluster = kmean.predict(data)
data['Cluster'] = cluster

# using my algorithm (chi method) to find the best 5 features
y = result['Life expectancy at birth (years)']
x = data
new = SelectKBest(chi2, k=3).fit_transform(x,y)
new = pd.DataFrame(new)
a = new
b = classlabel
#T rain the knn model and produce the accuracy socre
X_train, X_test, y_train, y_test = train_test_split(new,classlabel, train_size=(2/3), test_size=(1/3), random_state=100)
new = neighbors.KNeighborsClassifier(n_neighbors=5)
new.fit(X_train, y_train)
new_pred=new.predict(X_test)
print("Accuracy of feature engineering: "+ str(round(accuracy_score(y_test, new_pred)*100,3))+"%")

################# PCA #############################

#sperating the source and target into two table
pcakeys = [str(result) for result in pcadata.keys()]
pcafeacture = pcadata[pcakeys[1:]]
pcatarget = pcadata[pcakeys[0]]

#pca normalization, since pca can be incflacted by scale
idf=pd.DataFrame(imp.fit_transform(pcafeacture))
idf.columns=pcafeacture.columns
idf.index=pcafeacture.index
pcafeacture = idf
scaler = preprocessing.StandardScaler().fit(pcafeacture)
pcafeacture=scaler.transform(pcafeacture)

#using the library function to find out the best 4 feacutres
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(pcafeacture)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4'])


#Train the knn model and produce the accuracy socr
X_train, X_test, y_train, y_test = train_test_split(principalDf,pcatarget, train_size=(2/3), test_size=(1/3), random_state=100)
pcaresult = neighbors.KNeighborsClassifier(n_neighbors=5)
pcaresult.fit(X_train, y_train)
pcaresult_pred=pcaresult.predict(X_test)
print("Accuracy of PCA: "+ str(round(accuracy_score(y_test, pcaresult_pred)*100,3))+"%")




################ first 4 #########################
#seprating the table into feacture and target
gg = pcatarge = top4.iloc[:,0]
top4 = top4.iloc[:,1:5]

#sperate the table into traning and testing set
X_train, X_test, y_train, y_test = train_test_split(top4,gg, train_size=(2/3), test_size=(1/3))

#impute X train
imp = SimpleImputer(missing_values = np.nan, strategy="median")  
tdf=pd.DataFrame(imp.fit_transform(X_train))
tdf.columns=X_train.columns
tdf.index=X_train.index
X_train = tdf

#impute X test
imp = SimpleImputer(missing_values = np.nan, strategy="median")  
tdf=pd.DataFrame(imp.fit_transform(X_test))
tdf.columns=X_test.columns
tdf.index=X_test.index
X_test = tdf

#scale up
scaler = preprocessing.StandardScaler().fit(X_train)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

#apply knn-5 model and produce the accuracy score
top4 = neighbors.KNeighborsClassifier(n_neighbors=5)
top4.fit(X_train, y_train)
top4_pred=top4.predict(X_test)
print("Accuracy of first four features: "+ str(round(accuracy_score(y_test, top4_pred)*100,3))+"%")