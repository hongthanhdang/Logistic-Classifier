import numpy as np 
from preprocessing_image import load_image,HOG
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os 

METRO_IMAGES_DIR='/Users/thanhdang/repos/python/CV/LogisticClassfier/metropolitian/'
COUNTRY_IMAGES_DIR='/Users/thanhdang/repos/python/CV/LogisticClassfier/metropolitian/'
def train(metro_images_dir,country_images_dir):
    # load image
    metro_images=load_image(METRO_IMAGES_DIR)
    country_images=load_image(COUNTRY_IMAGES_DIR)
    # extract HOG features
    metro_fd=HOG(metro_images)
    country_fd=HOG(country_images)
    # train logistic regression
    X=np.concatenate((metro_fd,country_fd),axis=0)
    y=np.array([0]*metro_fd.shape[0]+[1]*country_fd.shape[0])
    print('X: ',X.shape)
    print('y: ',y.shape)
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=.2,random_state=1)
    # logistic_clf=LogisticRegression(penalty='l2',max_iter=1000,C=0.1,verbose=1,solver='liblinear')
    # logistic_clf.fit(X_train,y_train)
    # y_train_pred=logistic_clf.predict(X_train)
    svm_clf=LinearSVC(verbose=1,C=0.1)
    svm_clf.fit(X_train,y_train)
    y_train_pred=svm_clf.predict(X_train)
    print("Train accuracy: ",accuracy_score(y_train,y_train_pred))
    y_val_pred=svm_clf.predict(X_val)
    print("Validation accuracy: ",accuracy_score(y_val,y_val_pred))
def main():
    train(METRO_IMAGES_DIR,COUNTRY_IMAGES_DIR)
if __name__=="__main__":
    main()

