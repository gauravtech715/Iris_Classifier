from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#Loading Datasets
iris=datasets.load_iris()

#Printing Description and features
print(iris.DESCR)
features=iris.data
label=iris.target
print(features[0],label[0])

#Training the classifier
clf=KNeighborsClassifier()
clf.fit(features,label)

predict=clf.predict([[1,1,1,1]])
print(predict)