import streamlit as st
##
import numpy as np
import pandas as pd
import sweetviz as sv
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_breast_cancer


##---------------

##---------------
#---------------------------------#
# Page layout
st.set_page_config(page_title='The Machine Learning App',layout='wide')

#---------------------------------#
# Model the building
def build_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)
    
    st.markdown('**Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)
############################################


    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)

    st.subheader('Model Random Forest Classifier')

    st.markdown('**Training set**')
    Y_pred_train = rf.predict(X_train)
    #st.write('Confusion Matrix:')
    #st.info( confusion_matrix(Y_train, Y_pred_train) )

    st.write('Accuracy Training:')
    st.info( accuracy_score(Y_train, Y_pred_train) )

    st.markdown('** Test set**')



    Y_pred_test = rf.predict(X_test)
    st.write('Classification Report:')
    st.info(  classification_report(Y_test, Y_pred_test) )

    st.write('Accuracy Testing:')
    st.info( accuracy_score(Y_test, Y_pred_test) )




    st.write('Confusion Matrix:')
    cm1 = confusion_matrix(Y_test, Y_pred_test)
    st.write(cm1)
    
    tn1 = cm1[0,0]
    fp1 = cm1[0,1]
    tp1 = cm1[1,1]
    fn1 = cm1[1,0]

    total = tn1 + fp1 + tp1 + fn1
    real_positive = tp1 + fn1
    real_negative = tn1 + fp1


    accuracy  = (tp1 + tn1) / total * 100 # Accuracy Rate
    precision = tp1 / (tp1 + fp1) * 100# Positive Predictive Value
    recall    = tp1 / (tp1 + fn1)* 100 # True Positive Rate
    f1score  = 2 * precision * recall / (precision + recall)* 100
    specificity = tn1 / (tn1 + fp1)* 100 # True Negative Rate
    error_rate = (fp1 + fn1) / total * 100# Missclassification Rate
    prevalence = real_positive / total* 100
    miss_rate = fn1 / real_positive* 100 # False Negative Rate
    fall_out = fp1 / real_negative* 100 # False Positive Rate

    st.write('Accuracy:',accuracy)
    st.write('Precision:', precision)
    st.write('Recall:', recall)
    st.write('F1-score:', f1score)
    st.write('Specificity:', specificity)
    st.write('Error Rate:', error_rate)
    st.write('Prevalence:', prevalence)
    st.write('Miss Rate:', miss_rate)
    st.write('Fall out:', fall_out)





###############################################################

###############################################################

    
    xgboost = XGBClassifier()


    xgboost.fit(X_train, Y_train)

    st.subheader('Model Xgboost Classifier')

    st.markdown('**Training set**')
    Y_pred_train = xgboost.predict(X_train)
    #st.write('Confusion Matrix:')
    #st.info( confusion_matrix(Y_train, Y_pred_train) )

    st.write('Accuracy Training:')
    st.info( accuracy_score(Y_train, Y_pred_train) )

    st.markdown('** Test set**')



    Y_pred_test = xgboost.predict(X_test)
    st.write('Classification Report:')
    st.info(  classification_report(Y_test, Y_pred_test) )

    st.write('Accuracy Testing:')
    st.info( accuracy_score(Y_test, Y_pred_test) )




    st.write('Confusion Matrix:')
    cm2 = confusion_matrix(Y_test, Y_pred_test)
    st.write(cm2)
    
    tn2 = cm2[0,0]
    fp2 = cm2[0,1]
    tp2 = cm2[1,1]
    fn2 = cm2[1,0]

    total = tn2 + fp2 + tp2 + fn2
    real_positive = tp2 + fn2
    real_negative = tn2 + fp2


    accuracy  = (tp2 + tn2) / total * 100 # Accuracy Rate
    precision = tp2 / (tp2 + fp2) * 100# Positive Predictive Value
    recall    = tp2 / (tp2 + fn2)* 100 # True Positive Rate
    f1score  = 2 * precision * recall / (precision + recall)* 100
    specificity = tn2 / (tn2 + fp2)* 100 # True Negative Rate
    error_rate = (fp2 + fn2) / total * 100# Missclassification Rate
    prevalence = real_positive / total* 100
    miss_rate = fn2 / real_positive* 100 # False Negative Rate
    fall_out = fp2 / real_negative* 100 # False Positive Rate

    st.write('Accuracy:',accuracy)
    st.write('Precision:', precision)
    st.write('Recall:', recall)
    st.write('F1-score:', f1score)
    st.write('Specificity:', specificity)
    st.write('Error Rate:', error_rate)
    st.write('Prevalence:', prevalence)
    st.write('Miss Rate:', miss_rate)
    st.write('Fall out:', fall_out)




    st.subheader('')








##############################

    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)

    st.subheader('Model K-Neighbors Classifier')

    st.markdown('**Training set**')
    Y_pred_train = knn.predict(X_train)
    #st.write('Confusion Matrix:')
    #st.info( confusion_matrix(Y_train, Y_pred_train) )

    st.write('Accuracy Training:')
    st.info( accuracy_score(Y_train, Y_pred_train) )

    st.markdown('** Test set**')



    Y_pred_test = knn.predict(X_test)
    st.write('Classification Report:')
    st.info(  classification_report(Y_test, Y_pred_test) )

    st.write('Accuracy Testing:')
    st.info( accuracy_score(Y_test, Y_pred_test) )




    st.write('Confusion Matrix:')
    cm3 = confusion_matrix(Y_test, Y_pred_test)
    st.write(cm3)
    
    tn3 = cm3[0,0]
    fp3 = cm3[0,1]
    tp3 = cm3[1,1]
    fn3 = cm3[1,0]

    total = tn3 + fp3 + tp3 + fn3
    real_positive = tp3 + fn3
    real_negative = tn3 + fp3


    accuracy  = (tp3 + tn3) / total * 100 # Accuracy Rate
    precision = tp3 / (tp3 + fp3) * 100# Positive Predictive Value
    recall    = tp3 / (tp3 + fn3)* 100 # True Positive Rate
    f1score  = 2 * precision * recall / (precision + recall)* 100
    specificity = tn3 / (tn3 + fp3)* 100 # True Negative Rate
    error_rate = (fp3 + fn3) / total * 100# Missclassification Rate
    prevalence = real_positive / total* 100
    miss_rate = fn3 / real_positive* 100 # False Negative Rate
    fall_out = fp3 / real_negative* 100 # False Positive Rate

    st.write('Accuracy:',accuracy)
    st.write('Precision:', precision)
    st.write('Recall:', recall)
    st.write('F1-score:', f1score)
    st.write('Specificity:', specificity)
    st.write('Error Rate:', error_rate)
    st.write('Prevalence:', prevalence)
    st.write('Miss Rate:', miss_rate)
    st.write('Fall out:', fall_out)


###########################

##############################

    Ada = AdaBoostClassifier()
    Ada.fit(X_train, Y_train)

    st.subheader('Model AdaBoost Classifier ')

    st.markdown('**Training set**')
    Y_pred_train = Ada.predict(X_train)
    #st.write('Confusion Matrix:')
    #st.info( confusion_matrix(Y_train, Y_pred_train) )

    st.write('Accuracy Training:')
    st.info( accuracy_score(Y_train, Y_pred_train) )

    st.markdown('** Test set**')



    
    st.write('Classification Report:')
    st.info(  classification_report(Y_test, Y_pred_test) )
    Y_pred_test = Ada.predict(X_test)
    st.write('Accuracy Testing:')
    st.info( accuracy_score(Y_test, Y_pred_test) )




    st.write('Confusion Matrix:')
    cm4 = confusion_matrix(Y_test, Y_pred_test)
    st.write(cm4)
    
    tn4 = cm4[0,0]
    fp4 = cm4[0,1]
    tp4 = cm4[1,1]
    fn4 = cm4[1,0]

    total = tn4 + fp4 + tp4 + fn4
    real_positive = tp4 + fn4
    real_negative = tn4 + fp4


    accuracy  = (tp4 + tn4) / total * 100 # Accuracy Rate
    precision = tp4 / (tp4 + fp4)  # Positive Predictive Value
    recall    = tp4 / (tp4 + fn4)* 100 # True Positive Rate
    f1score  = 2 * precision * recall / (precision + recall)* 100
    specificity = tn4 / (tn4 + fp4)* 100 # True Negative Rate
    error_rate = (fp4 + fn4) / total * 100# Missclassification Rate
    prevalence = real_positive / total* 100
    miss_rate = fn4 / real_positive* 100 # False Negative Rate
    fall_out = fp4 / real_negative* 100 # False Positive Rate

    st.write('Accuracy:',accuracy)
    st.write('Precision:', precision)
    st.write('Recall:', recall)
    st.write('F1-score:', f1score)
    st.write('Specificity:', specificity)
    st.write('Error Rate:', error_rate)
    st.write('Prevalence:', prevalence)
    st.write('Miss Rate:', miss_rate)
    st.write('Fall out:', fall_out)


###########################

    catbo = CatBoostClassifier()


    catbo.fit(X_train, Y_train)

    st.subheader('Model CatBoost Classifier')

    st.markdown('**Training set**')
    Y_pred_train = catbo.predict(X_train)
    #st.write('Confusion Matrix:')
    #st.info( confusion_matrix(Y_train, Y_pred_train) )

    st.write('Accuracy Training:')
    st.info( accuracy_score(Y_train, Y_pred_train) )

    st.markdown('** Test set**')



    Y_pred_test = catbo.predict(X_test)
    st.write('Classification Report:')
    st.info(  classification_report(Y_test, Y_pred_test) )

    st.write('Accuracy Testing:')
    st.info( accuracy_score(Y_test, Y_pred_test) )




    st.write('Confusion Matrix:')
    cm5 = confusion_matrix(Y_test, Y_pred_test)
    st.write(cm5)
    
    tn5 = cm5[0,0]
    fp5 = cm5[0,1]
    tp5 = cm5[1,1]
    fn5 = cm5[1,0]

    total = tn5 + fp5 + tp5 + fn5
    real_positive = tp5 + fn5
    real_negative = tn5 + fp5


    accuracy  = (tp5 + tn5) / total * 100 # Accuracy Rate
    precision = tp5 / (tp5 + fp5) * 100# Positive Predictive Value
    recall    = tp5 / (tp5 + fn5)* 100 # True Positive Rate
    f1score  = 2 * precision * recall / (precision + recall)* 100
    specificity = tn5 / (tn5 + fp5)* 100 # True Negative Rate
    error_rate = (fp5 + fn5) / total * 100# Missclassification Rate
    prevalence = real_positive / total* 100
    miss_rate = fn5 / real_positive* 100 # False Negative Rate
    fall_out = fp5 / real_negative* 100 # False Positive Rate

    st.write('Accuracy:',accuracy)
    st.write('Precision:', precision)
    st.write('Recall:', recall)
    st.write('F1-score:', f1score)
    st.write('Specificity:', specificity)
    st.write('Error Rate:', error_rate)
    st.write('Prevalence:', prevalence)
    st.write('Miss Rate:', miss_rate)
    st.write('Fall out:', fall_out)




    st.subheader('')








##############################

 
    drbc = GradientBoostingClassifier()


    drbc.fit(X_train, Y_train)

    st.subheader('Model GradientBoosting Classifier')

    st.markdown('**Training set**')
    Y_pred_train = drbc.predict(X_train)
    #st.write('Confusion Matrix:')
    #st.info( confusion_matrix(Y_train, Y_pred_train) )

    st.write('Accuracy Training:')
    st.info( accuracy_score(Y_train, Y_pred_train) )

    st.markdown('** Test set**')



    Y_pred_test = drbc.predict(X_test)
    st.write('Classification Report:')
    st.info(  classification_report(Y_test, Y_pred_test) )

    st.write('Accuracy Testing:')
    st.info( accuracy_score(Y_test, Y_pred_test) )




    st.write('Confusion Matrix:')
    cm6 = confusion_matrix(Y_test, Y_pred_test)
    st.write(cm6)
    
    tn6 = cm6[0,0]
    fp6 = cm6[0,1]
    tp6 = cm6[1,1]
    fn6 = cm6[1,0]

    total = tn6 + fp6 + tp6 + fn6
    real_positive = tp6 + fn6
    real_negative = tn6 + fp6


    accuracy  = (tp6 + tn6) / total * 100 # Accuracy Rate
    precision = tp6 / (tp6 + fp6) * 100# Positive Predictive Value
    recall    = tp6 / (tp6 + fn6)* 100 # True Positive Rate
    f1score  = 2 * precision * recall / (precision + recall)* 100
    specificity = tn6 / (tn6 + fp6)* 100 # True Negative Rate
    error_rate = (fp6 + fn6) / total * 100# Missclassification Rate
    prevalence = real_positive / total* 100
    miss_rate = fn6 / real_positive* 100 # False Negative Rate
    fall_out = fp6 / real_negative* 100 # False Positive Rate

    st.write('Accuracy:',accuracy)
    st.write('Precision:', precision)
    st.write('Recall:', recall)
    st.write('F1-score:', f1score)
    st.write('Specificity:', specificity)
    st.write('Error Rate:', error_rate)
    st.write('Prevalence:', prevalence)
    st.write('Miss Rate:', miss_rate)
    st.write('Fall out:', fall_out)




    st.subheader('')

###############################
#################


###############################
#################


    



#---------------------------------#
st.write(""" 
    # *App Ammar *


In this implementation to create a Classifier model and Exploratory Data Analysis (EDA)  using 6 modeles :
* 1-Model Random Forest Classifier.
* 2- Model XGB Classifier.
* 3-Model K-Neighbors Classifier.
* 4-Model Ada Boost Classifier.
* 5-Model Cat Boost Classifier.
* 6-Model Gradient Boosting Classifier.


""")


#---------------------------------#

with st.sidebar.header(' Upload your File CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/yasserhessein/deep-learning-classification-mammographic-mass/main/Cleaned_data.csv)
""")

with st.sidebar.header(' Split the Dataset'):
    split_size = st.sidebar.slider('Training Set', 10, 90, 80, 5)

  

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('All the Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**  Dataset**')
    st.write(df)


    st.markdown('**Tail **')
    st.write(df.tail())


    st.markdown('**Shape **')
    st.write(df.shape)



    st.markdown('**Check Miss Dataset **')
    st.write(df.isnull().sum())



    st.markdown('**Describe the Dataset**')
    st.write(df.describe().T)
    st.markdown('**Correlation the Dataset**')
    st.write(df.corr().T)

    Tr_report1 = sv.analyze(df)
    st.write(Tr_report1)
    st.write(Tr_report1.show_notebook(w="80%", h="full"))
    st.header(Tr_report1.show_html('Tr_report1.html'))
    #st.header(sns.heatmap(df.corr(), annot=True, fmt='.0%'))

    ##================





  ##@@@@@@@@@@@@@@###########


    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Diabetes dataset
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        Y = pd.Series(data.target, name='Class')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Breast Cancer dataset is used as the example.')
        st.write(df.head(5))




        st.markdown('**Tail **')
        st.write(df.tail())


        st.markdown('**Shape **')
        st.write(df.shape)



        st.markdown('**Check Miss Dataset **')
        st.write(df.isnull().sum())



        st.markdown('**Describe the Dataset**')
        st.write(df.describe().T)
        st.markdown('**Correlation the Dataset**')
        st.write(df.corr().T)

        Tr_report1 = sv.analyze(df)

        st.header(Tr_report1.show_html('Tr_report1.html'))


        build_model(df)


##st.header("An owl")
##st.image("https://static.streamlit.io/examples/owl.jpg")

st.image('mlll.gif')

st.text(' The Machine Learning App By Yasir Huusein Shakir') 

st.text(' App Ammar version 0.0.1') 
st.text(' 7-23-2021') 



st.text('How to reach me ?')
st.text('Emails:')
st.text('Uniten : pe20911@uniten.edu.my')
st.text('Yahoo : yasserhesseinshakir@yahoo.com')
st.text('Kaggle : https://www.kaggle.com/yasserhessein')
st.text('GitHub : https://github.com/yasserhessein')
st.text('linkedin : https://www.linkedin.com/in/yasir-hussein-314a65201/')

## Source :
##(https://scikit-learn.org/stable/)
##(https://id.heroku.com/)
##(https://streamlit.io/)
