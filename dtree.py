#Import all the necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier    
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report,plot_confusion_matrix
#from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings(True)


df=pd.read_csv('https://raw.githubusercontent.com/kaurpreet578/D_tree/main/diabetes.csv')

X= df[df.columns[:-1]]
y=df['Outcome']

# Split the dataset into train and test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 30)

#Resampling the data
# rus= RandomUnderSampler(sampling_strategy='not minority',random_state=30)
# X_train_u,y_train_u= rus.fit_resample(X_train,y_train)


#Adding a header
st.title('''
	A web app to predict if a person has diabetes or not, using DecisionTreeClassifier
	''')
st.sidebar.header('Menu')
st.sidebar.subheader('Select the inputs')

#Creating sliders in the sidebar
glucose=st.sidebar.slider('Glucose',int(df['Glucose'].min()),int(df['Glucose'].max()))
bp=st.sidebar.slider('BloodPressure',int(df['BloodPressure'].min()),int(df['BloodPressure'].max()))
preg=st.sidebar.slider('Pregnancies',int(df['Pregnancies'].min()),int(df['Pregnancies'].max()))
skin_thk=st.sidebar.slider('SkinThickness',int(df['SkinThickness'].min()),int(df['SkinThickness'].max()))
insulin=st.sidebar.slider('Insulin',int(df['Insulin'].min()),int(df['Insulin'].max()))
bmi=st.sidebar.slider('BMI',float(df['BMI'].min()),float(df['BMI'].max()))
diab=st.sidebar.slider('DiabetesPedigreeFunction',float(df['DiabetesPedigreeFunction'].min()),float(df['DiabetesPedigreeFunction'].max()))
age=st.sidebar.slider('Age',int(df['Age'].min()),int(df['Age'].max()))

#Setting deprecation warning to False to avoid interruption
st.set_option('deprecation.showPyplotGlobalUse', False)

#Defining the prediction function
def prediction(glucose,bp,preg,skin_thk,insulin,bmi,diab,age):
	
	dtree_clf_u= DecisionTreeClassifier(random_state = 42,criterion='entropy',max_depth=6,
	min_samples_leaf=7,min_samples_split=5)
	dtree_clf_u.fit(X_train, y_train)
	y_train_pred_u = dtree_clf_u.predict(X_train)
	y_test_pred_u = dtree_clf_u.predict(X_test)

	#Calculating f1-score for Train and Test data
	train_score= metrics.f1_score(y_train, y_train_pred_u,average=None)
	test_score= metrics.f1_score(y_test, y_test_pred_u,average=None)

	train_acc=metrics.accuracy_score(y_train, y_train_pred_u)
	test_acc= metrics.accuracy_score(y_test, y_test_pred_u)

	#Prediction on the given set of input values
	pred= dtree_clf_u.predict([[glucose,bp,preg,skin_thk,insulin,bmi,diab,age]])
	pred=pred[0]
	if pred==0:
		pred='Not having diabetes'
	elif pred==1:
		pred= 'having diabetes'
	#Creating confusion matrix	
	cm=plot_confusion_matrix(dtree_clf_u,X_test,y_test,values_format='d')
	st.pyplot()
	return pred, train_score,test_score,train_acc,test_acc

#Button to run the classification
if st.sidebar.button('Classify'):


	predict,train_score,test_score,train_acc,test_acc=prediction(glucose,bp,preg,skin_thk,insulin,bmi,diab,age)
	st.header('Model deployment results')
	st.write(f'The person is {predict}')

	st.subheader('Evaluation results')
	st.write(f'f1-score for the train-set is {train_score}')
	st.write(f'f1-score for the test-set is {test_score}')

	st.success(f"Accuracy on the train set: {train_acc:.4f}\n")
	st.success(f"Accuracy on the test set: {test_acc:.4f}")
	

	
	






