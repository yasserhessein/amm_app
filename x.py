import streamlit as st

##

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sweetviz as sv

from sklearn.metrics import classification_report

##
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_breast_cancer


##---------------

##---------------
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App',layout='wide')

#---------------------------------#
# Model building
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



    rf = RandomForestClassifier(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        #criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        #oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs)
    rf.fit(X_train, Y_train)

    st.subheader('Model Performance')

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
    cm = confusion_matrix(Y_test, Y_pred_test)
    st.write(cm)
    
    tn = cm[0,0]
    fp = cm[0,1]
    tp = cm[1,1]
    fn = cm[1,0]

    total = tn + fp + tp + fn
    real_positive = tp + fn
    real_negative = tn + fp


    accuracy  = (tp + tn) / total * 100 # Accuracy Rate
    precision = tp / (tp + fp) * 100# Positive Predictive Value
    recall    = tp / (tp + fn)* 100 # True Positive Rate
    f1score  = 2 * precision * recall / (precision + recall)* 100
    specificity = tn / (tn + fp)* 100 # True Negative Rate
    error_rate = (fp + fn) / total * 100# Missclassification Rate
    prevalence = real_positive / total* 100
    miss_rate = fn / real_positive* 100 # False Negative Rate
    fall_out = fp / real_negative* 100 # False Positive Rate

    st.write('Accuracy:',accuracy)
    st.write('Precision:', precision)
    st.write('Recall:', recall)
    st.write('F1-score:', f1score)
    st.write('Specificity:', specificity)
    st.write('Error Rate:', error_rate)
    st.write('Prevalence:', prevalence)
    st.write('Miss Rate:', miss_rate)
    st.write('Fall out:', fall_out)




    st.subheader('Model Parameters')
    st.write(rf.get_params())




#---------------------------------#
st.write("""
# The Machine Learning App By Yasir Huusein Shakir 

Exploratory Data Analysis (EDA) and RandomForestClassifier(RFC) functions are utilised in this implementation to create a Classifier model using 

the Random Forest method try adjusting the hyperparameters!

""")


#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header(' Upload your File CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/yasserhessein/deep-learning-classification-mammographic-mass/main/Cleaned_data.csv)
""")

# Sidebar - Specify parameter settings
with st.sidebar.header(' Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader(' Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['gini', 'entropy'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader(' General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
   
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])


    
    parameter_bootstrap = st.sidebar.select_slider('warm_start reuse the solution of the previous call to fit and add more estimators to the ensemble (warm_start)', options=[True, False])
    parameter_bootstrap = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate (oob_score)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider(' the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_criterion = st.sidebar.select_slider('Weights associated with classes (class_weight)', options=['balanced', 'balanced_subsample'])
    

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader(' Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('** Glimpse of dataset**')
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
    #st.write(Tr_report1)
    #st.write(Tr_report1.show_notebook(w="80%", h="full"))
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

st.text('How to reach me ?')
st.text('Emails:')
st.text('Uniten : pe20911@uniten.edu.my')
st.text('Yahoo : yasserhesseinshakir@yahoo.com')
st.text('Kaggle : https://www.kaggle.com/yasserhessein')
st.text('GitHub : https://github.com/yasserhessein')
st.text('linkedin : https://www.linkedin.com/in/yasir-hussein-314a65201/')