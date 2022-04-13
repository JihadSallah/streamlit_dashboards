#import libraries
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#App ki heading
st.write("""
# Explore different ML models and Datasets
Let's see which one is the best
""")

# Data sets
dataset_name=st.sidebar.selectbox('Select Dataset',("Iris","Breast Cancer","Wine"))

# Selection of ML classifiers
classifier_name=st.sidebar.selectbox("Select classifier",("KNN","SVM",'Randm forest'))

#Now by defining datasets for loading
def get_dataset(dataset_name):
    data=None
    if dataset_name=="Iris":
        data=datasets.load_iris()
    elif dataset_name=="Wine":
        data=datasets.load_wine()
    else: 
        data=datasets.load_breast_cancer()
    x=data.data
    y=data.target
    return x,y
#Now calling the function defined above
X,y=get_dataset(dataset_name)

# finding the shape of the data set
st.write("Shape of dataset:",X.shape)
st.write("Number of classes",len(np.unique(y)))

#now we are going to use classifier parameters
def add_parameter_ui(classifier_name):
    params=dict() #creating an empty dictionary
    if classifier_name=="SVM":
        C=st.sidebar.slider("C",0.01,10.0)
        params['C']=C #its the degree of correct correlation
    elif classifier_name=="KNN":
        K=st.sidebar.slider("K",1,15)
        params['K']=K #Its the number of nearest neighbour
    else:
        max_depth=st.sidebar.slider("max_depth",2,15)
        params["max_depth"]=max_depth #depth of every tree that grows in random forest
        n_estimators=st.sidebar.slider("n_estimators",1,100)
        params["n_estimators"]=n_estimators #number of trees
    return params
#Now calling the params
params=add_parameter_ui(classifier_name)

#now making classifiers on the classifier_name and params
def get_classifier(classifier_name,params):
    clf=None
    if classifier_name=="SVM":
        clf=SVC(C=params["C"])
    elif classifier_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])
    else:
        clf=RandomForestClassifier(n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],random_state=1234)
    return clf
#now calling the function
clf=get_classifier(classifier_name,params)

#Splitting datasets in test and train
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)

# Now training the classifier
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

#Model ka accuracy score check kr lain
acc=accuracy_score(y_test,y_pred)
st.write(f"Classifier={classifier_name}")
st.write(f"Accuracy=",acc)

#Plotting the data set
pca=PCA(2)
X_projected=pca.fit_transform(X)

#Now changing the data by slicing on the scale of 0 to 1
x1=X_projected[:,0]
x2=X_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")

plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()

#plt.show()
st.pyplot(fig)









