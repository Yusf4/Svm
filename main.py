import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data,labels=make_classification(
    n_samples=200,n_features=2,n_classes=2,n_informative=2,
    n_redundant=0,random_state=42,n_clusters_per_class=1
)
#split the dataset
X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.3,random_state=42)

#initialize and train the svm model
svm_model=SVC(kernel='linear',random_state=42)
svm_model.fit(X_train,y_train)

#make predictions
predictions=svm_model.predict(X_test)

#evaluate the model
accuracy=accuracy_score(y_test,predictions)
print(f" accuracy:{accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test,predictions))


def plot_decision_boundary(clf,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:, 1].min()-1,X[:, 1].max()+ 1
    xx,yy =np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))

    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)

    plt.contourf(xx,yy,Z,alpha=0.8,cmap=plt.cm.Paired)
    plt.scatter(X[:, 0],X[:, 1],c=y,edgecolors='k',cmap=plt.cm.Paired)
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.show()
plot_decision_boundary(svm_model,X_test,y_test)