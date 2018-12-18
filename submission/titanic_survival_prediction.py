"""
CS 286 Data Analysis and Prediction Project
Analyszing titanic survival data and predicting titanic survival using SVC classifier

Usage: run `python titanic_survival_prediction.py`

Project Group : 3
Authors: Saketh Saxena, Amer Rez, Anand V., Vedashree

Known Issue:
    Requires the latest version of sklearn - 0.20.0
    
Last updated:12/18/2018
"""

# imports
import sys
try:
    import numpy as np
except:
    print("Please install numpy ")
    print("Exiting program")
    sys.exit(0)

try:
    import pandas as pd
except:
    print("Please install pandas")
    print("Exiting program")
    sys.exit(0)

try:
    import matplotlib.pyplot as plt
except:
    print("Please install matplotlib")
    print("Exiting program")
    sys.exit(0)

try:
    import seaborn as sns
except:
    print("Please install seaborn")
    print("Exiting program")
    sys.exit(0)

try:
    from sklearn.impute import SimpleImputer
except:
    print("Please install the latest version of sklearn-0.20.0")
    print("Exiting program")
    sys.exit(0)

try:
    from sklearn.svm import SVC
except:
    print("Please install sklearn.svm")
    print("Exiting program")
    sys.exit(0)

try:
    from sklearn.model_selection import cross_validate
except:
    print("Please install sklearn.model_selection")
    print("Exiting program")
    sys.exit(0)

try:
    from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
except:
    print("Please install sklearn.metrics")
    print("Exiting program")
    sys.exit(0)

    
# Global variables    
__PREDICTOR_VARIABLES__1 = [1,2,3,'Age', 'SibSp', 'Parch', 'Fare', 'female', 'C', 'Q', 'S']
__PREDICTOR_VARIABLES__2 = ['Fare', 'female']
__PREDICTOR_VARIABLES__3 = [3, 'female']

##################################
# Helper functions              ##
##################################

# inputs: csv file  name;
# returns : dataframe from the csv file;
# source : CS 286 sample code
def read_file(filename):
    try:
        print("Reading the data set named: ", filename)
        df = pd.read_csv(filename)
    except IOError:
        raise IOError("Problems locating or opening the dataset named " + filename)
    print("Completed reading ", filename)
    print()
    return df
### end of read_file

# inputs: dataframe; name of a column to be one hot encoded;
# returns : dataframe with the specific column encoded with one hot encoding
def one_hot_enc(df, column_name):
    one_hot = pd.get_dummies(df[column_name])
    df = df.drop(column_name,axis = 1)
    df = df.join(one_hot)
    return df
### end of one_hot_enc

# inputs : predictor variable list; cross validation scores;
# returns :prints the results of cross fold validation
# print list : cv scores list, mean accuracy, standard deviation of accuracy, precision, recall, f1 score, score time, fit time
def print_cv_results(model_no,__PREDICTOR_VARIABLES__,scores):
    # printing cross validation scores    


    print("Cross validation scores for model "+str(model_no)+" with predictors",__PREDICTOR_VARIABLES__)
    print("####")

    print("Cross validation Accuracy scores : ",scores["test_accuracy"])
    print("Mean accuracy : "+str(scores["test_accuracy"].mean())+" Standard deviation : "+str(scores["test_accuracy"].std()*2.0))    

    print("Cross validation Precision scores : ",scores["test_precision"])
    print("Mean precision : "+str(scores["test_precision"].mean())+" Standard deviation : "+str(scores["test_precision"].std()*2.0))    
    
    print("Cross validation Recall scores : ",scores["test_recall"])
    print("Mean Recall : "+str(scores["test_recall"].mean())+" Standard deviation : "+str(scores["test_recall"].std()*2.0))    

    print("Cross validation f1 scores : ",scores["test_f1"])
    print("Mean f1 : "+str(scores["test_f1"].mean())+" Standard deviation : "+str(scores["test_f1"].std()*2.0))    
    
    print("Score time: ",scores["score_time"])
    print("Mean Score time : "+str(scores["score_time"].mean())+" Standard deviation : "+str(scores["score_time"].std()*2.0))    
    
    print("fit time: ",scores["fit_time"])     
    print("Mean fit time : "+str(scores["fit_time"].mean())+" Standard deviation : "+str(scores["fit_time"].std()*2.0))    
    print("####")
    print()
    return
### end of print_cv_results
    
# inputs: model_no; actual_class df; predicted_class df;
# returns: prints the accuracy, precision. recall and confusion matrix of the test on real data
# print list : accuracyy %, precision, recall and confusion matrix
def print_test_results(model_no, actual_class,predicted_class):
    # printing model performance scores based on test data

    print()
    print("Test Results for model no: ",model_no)
    print("Accuracy % of predictions on test data",accuracy_score(actual_class["Survived"], predicted_class, normalize=True, sample_weight=None)*100)    
    print("Precision of predictions on test data",precision_score(actual_class["Survived"], predicted_class,   average='macro'))    
    print("Recall of predictions on test data",recall_score(actual_class["Survived"], predicted_class,  average='macro'))  
    print("Confusion matrix :\n",confusion_matrix(actual_class["Survived"], predicted_class))
    print("--------------------------------------------------------------------------------------")
    return
### end of print_test_results

##################################
# End Helper functions          ##
##################################


###################################
# data analysis process functions #
###################################

# inputs : dataframe; threshold value;
# returns : prints the Pearson product-moment correlation coefficients between 
#     target vs predictor values
#     amongst the predictor values
def feature_correlation_analysis(df,threshold = 0.2):
    print("--------------------------------------------------------------------------------------")
    print("Feature correlation analysis using Pearson product-moment correlation coefficients with threshold "+str(threshold))
    # One hot encoding on feature Sex
    print("Performing one hot encoding on Sex column")
    df = one_hot_enc(df,"Sex")
    
    # One hot encoding on feature Embarked
    print("Performing one hot encoding on Embarked column")
    df = one_hot_enc(df,"Embarked")
    
    # One hot encoding on feature Pclass
    print("Performing one hot encoding on Pclass column")
    df = one_hot_enc(df,"Pclass")    
    
    # removing columns which do not impact the class
    selectedColumns = list(df.columns)
    selectedColumns.remove("PassengerId")
    selectedColumns.remove("Cabin")
    selectedColumns.remove("Name")
    selectedColumns.remove("Ticket")
    
    # computing Pearson product-moment correlation coefficients
    cm = np.corrcoef(df[selectedColumns].values.T) 
    

    ## print the features corelated with the target

    print("The features corelated with the target based on threshold " + str(threshold))
    for rowIndex in range(len(cm)):
        corrIndex = 0 # the target index
        if rowIndex != corrIndex:
            if (cm[rowIndex][corrIndex] > threshold or cm[rowIndex][corrIndex] < -threshold) :
                print(str(selectedColumns[rowIndex]) + " and " + str(selectedColumns[corrIndex]) + " are dependent")
    print()

    ## print the features corelated with each others
    print("The features corelated with each others based on threshold " + str(threshold))
    for rowIndex in range(1, len(cm)):
        for corrIndex in range(1, len(cm[rowIndex])):
            if rowIndex != corrIndex:
                if (cm[rowIndex][corrIndex] > threshold or cm[rowIndex][corrIndex] < -threshold):
                    print(str(selectedColumns[rowIndex]) + " and " + str(selectedColumns[corrIndex]) + " are dependent")
    print("Done")
    print("--------------------------------------------------------------------------------------")
    return
### end of feature_correlation_analysis
         
# inputs : dataframe to be visualized
# returns : plots freq bar graphs between target and predictors and correlation heatmap of all features
#           also prints descriptive statistics for each plot
def data_visualization(df):        
    print("--------------------------------------------------------------------------------------")
    print("Plotting heatmap to show correlation among features")
    # HeatMap using seborn
    plt.figure(1)
    sns.heatmap(df.corr(), 
            annot=True,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            vmax = .9,
            square=True)
    plt.title("Correlations among features", y = 1.03,fontsize = 20);
    
    
    
    
    colorList = ['#78C850',  # Grass
                        '#F08030',  # Fire
                        '#6890F0',  # Water
                        '#A8B820',  # Bug
                        '#A8A878',  # Normal
                        '#A040A0',  # Poison
                        '#F8D030',  # Electric
                        '#E0C068',  # Ground
                        '#EE99AC',  # Fairy
                        '#C03028',  # Fighting
                        '#F85888',  # Psychic
                        '#B8A038',  # Rock
                        '#705898',  # Ghost
                        '#98D8D8',  # Ice
                        '#7038F8',  # Dragon
                        ]
    print("--------------------------------------------------------------------------------------")                    
    print("Plotting sex Vs survived")
    # Percentage of Male and Female who survived
    plt.figure(3)
    print("Sex Feature Analysis")
    sns.barplot(x='Sex',y='Survived',data=df, palette= colorList).set_title("Survival Rate Vs Sex")
    print("Percentage of females who survived:", df["Survived"][df["Sex"] == "female"].value_counts(normalize = True)[1]*100)    
    print("Percentage of males who survived:", df["Survived"][df["Sex"] == "male"].value_counts(normalize = True)[1]*100)
    print("--------------------------------------------------------------------------------------")

    # Comparing the Pclass feature against Survived
    
    print("Plotting Pclass Vs survived")
    plt.figure(4)
    
    sns.barplot(x='Pclass',y='Survived',data=df, palette= colorList).set_title("Survival Rate Vs Class")
    print("Percentage of Pclass = 1 who survived:", df["Survived"][df["Pclass"] == 1].value_counts(normalize = True)[1]*100)
    print("Percentage of Pclass = 2 who survived:", df["Survived"][df["Pclass"] == 2].value_counts(normalize = True)[1]*100)    
    print("Percentage of Pclass = 3 who survived:", df["Survived"][df["Pclass"] == 3].value_counts(normalize = True)[1]*100)
    print("--------------------------------------------------------------------------------------")


    print("Plotting Parch vs survived")
    plt.figure(5)
    sns.barplot(x='Parch',y='Survived',data=df, palette= colorList).set_title("Survival Rate Vs Parch")
    # Comparing the Parch feature against Survived
    grouped_df = df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    print("\nPercentage of people who survived with following no of parents or children\n",grouped_df)
    print("--------------------------------------------------------------------------------------")


    # Comparing the Sibling  feature against Survived
    print("Plotting no of Siblings vs survived")
    plt.figure(6)
    sns.barplot(x='SibSp',y='Survived',data=df,palette= colorList).set_title("Survival Rate Vs SibSp")
    print("Percentage of SibSp = 0 who survived:", df["Survived"][df["SibSp"] == 0].value_counts(normalize = True)[1]*100)
    print("Percentage of SibSp = 1 who survived:", df["Survived"][df["SibSp"] == 1].value_counts(normalize = True)[1]*100)
    print("Percentage of SibSp = 2 who survived:", df["Survived"][df["SibSp"] == 2].value_counts(normalize = True)[1]*100)
    print("--------------------------------------------------------------------------------------")

    # Comparing the embarked  feature against Survived
    print("Plotting embarked vs survived")
    plt.figure(7)
    sns.barplot(x='Embarked',y='Survived',data=df,palette= colorList).set_title("Survival Rate Vs Embarked")
    print("Percentage of SibSp = 0 who survived:", df["Survived"][df["Embarked"] == "S"].value_counts(normalize = True)[1]*100)
    print("Percentage of SibSp = 1 who survived:", df["Survived"][df["Embarked"] == "Q"].value_counts(normalize = True)[1]*100)
    print("Percentage of SibSp = 2 who survived:", df["Survived"][df["Embarked"] == "C"].value_counts(normalize = True)[1]*100)
    print("--------------------------------------------------------------------------------------")
 
    print("Plotting age vs survived")
    # Survival  by Age 
    # sort the ages into logical categories    
    df["Age"] = df["Age"].fillna(-0.5)
    bins = [0, 5, 18, 35, 60, np.inf]
    labels = ['Baby', 'Child', 'Teenager', 'Adult', 'Senior']
    df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)
    plt.figure(8)
    #draw a bar plot of Age vs. survival
    sns.barplot(x="AgeGroup", y="Survived", data=df, palette= colorList).set_title("Survival Rate Vs Age groups")
    print("--------------------------------------------------------------------------------------")
    plt.show()
    print("Data Visualization Done")
    print()    
    return
### end of data_visualization

# inputs: dataframe to be cleaned
# returns : cleaned dataframe 
#           drops column 'Cabin'
#           drops records with nan in 'Embarked'
#           imputes age column with the median value
#           Performs one hot encoding on 'Sex'
#           Performs one hot encoding on 'Embarked'
#           Performs one hot encoding on 'Pclass'

def data_cleaning(df):
    
    print("--------------------------------------------------------------------------------------")
    print("Starting data cleaning process......")
    
    # Drop Cabin since many missing values
    print("Dropping column Cabin due to missing values")
    df = df.drop(columns='Cabin')
    
    print("Dropping rows with empty values for Embarked")
    df = df[pd.notnull(df['Embarked'])]

    # Imputing age column with median values for the column    
    print("Imputing age column with the median value")
    imp=SimpleImputer(strategy="median")
    imp.fit(df[["Age"]])
    df["Age"]=imp.transform(df[["Age"]]).ravel()    

    # One hot encoding on feature Sex
    print("Performing one hot encoding on Sex column")
    df = one_hot_enc(df,"Sex")
    
    # One hot encoding on feature Embarked
    print("Performing one hot encoding on Embarked column")
    df = one_hot_enc(df,"Embarked")
    
    # One hot encoding on feature Pclass
    print("Performing one hot encoding on Pclass column")
    df = one_hot_enc(df,"Pclass")

    print("Data cleaning done")        
    print("--------------------------------------------------------------------------------------")
    print()
    return df
### end of data_cleaning

# inputs : model_no,dataframe, list of predictor variables
# returns : trained model and scores
#           trains the SVC model 
#           Performs 5 fold cross validation on fitted model
def model(model_no,df,__PREDICTOR_VARIABLES__):
   
    print("--------------------------------------------------------------------------------------")    
    clf = SVC(C = 1, gamma = 'auto', class_weight=None, coef0=0.0, kernel = 'linear')
    print("Training SVC the model :",model_no)
    print("Predictor Variable List :", __PREDICTOR_VARIABLES__)
    clf.fit(df[__PREDICTOR_VARIABLES__], df["Survived"])
    print("Model "+str(model_no)+" training done!")
    
    print("Performing 5 Fold cross validation")
    scores = cross_validate(clf, df[__PREDICTOR_VARIABLES__], df["Survived"], cv=5,scoring=('accuracy', 'precision','recall','f1'),return_train_score=False)
    print("Model training and cross validation done")
    return clf,scores
### end of model



# inputs : test dataset dataframe; age_median, fare_median
# returns : cleaned test_df
#           imputes age and fare with median values from training data
#           performs one hot encoding on 'Sex', 'Embarked' and 'Pclass'
def clean_test_data(test_df,age_median,Fare_median):
    print("--------------------------------------------------------------------------------------")
    print("Cleaning test data")
    # data imputation for mmissing values in test set in fare and age columns
    print("Imputing age column of test set with the median value of training data")    
    test_df.Age.fillna(age_median,inplace=True)
    print("Imputing Fare column of test set with the median value of training data")    
    test_df.Fare.fillna(Fare_median,inplace=True)

    # One hot encoding on feature Sex in test dataset
    test_df = one_hot_enc(test_df,"Sex")
    # One hot encoding on feature Embarked in test dataset
    test_df = one_hot_enc(test_df,"Embarked")
    # One hot encoding on feature Pclass in test dataset
    test_df = one_hot_enc(test_df,"Pclass")
    print("Done")
    print("--------------------------------------------------------------------------------------")

    print()

    return test_df
### end of clean_test_data

# main function
# serves as the overall pipeline for all experiments 
# performs testing on the svc models
def main():
    
    print("#############################################################################################")
    print("Starting data analysis and prediction process for titanic survival prediction")
    # reading training file
    print("Reading training file")
    df = read_file("train.csv")
    
    # intial data analysis
    # printing results of feature correlation analysis with threshold - 0.2
    feature_correlation_analysis(df)
    # printing results of feature correlation analysis with threshold - 0.5
    feature_correlation_analysis(df,0.5)
    # printing results of feature correlation analysis with threshold - 0.8
    feature_correlation_analysis(df,0.8)
    

    # data visualization
    data_visualization(df)
    
    # data cleaning - pruning, imputation, one hot encoding
    cleaned_df = data_cleaning(df)
    print("df.shape before cleaning : ",df.shape)
    print("df.shape after cleaning : ",cleaned_df.shape)
        
    # extracting the median age and fare values of the training set
    age_median = df["Age"].median()    
    Fare_median = df["Fare"].median()    
    
    # reading in test data
    test_df = read_file("test.csv")
    
    # cleaning test_df
    test_df = clean_test_data(test_df,age_median,Fare_median)
    
    # reading in actual classes from gender_submission.csv
    actual_class = pd.read_csv("gender_submission.csv")
    
    #  Model 1:
    # training model and performing 5 fold cross validation
    #  on predictor list 1
    clf1, scores = model(1,cleaned_df,__PREDICTOR_VARIABLES__1)    
    # printing cv scores for model with first set of predictor variables
    print_cv_results(1,__PREDICTOR_VARIABLES__1,scores)    
    
    # testing the model 1 on real test set
    test_df1=test_df[__PREDICTOR_VARIABLES__1]        
    print("Testing Model 1 against test data")
    predicted_class = clf1.predict(test_df1)    
    print("Done")
    print_test_results(1,actual_class,predicted_class)
    
    #  Model 2:
    # training model and performing 5 fold cross validation
    #  on predictor list 2
    clf2, scores = model(2,cleaned_df,__PREDICTOR_VARIABLES__2)        
    # printing cv scores for model with second set of predictor variables
    print_cv_results(2,__PREDICTOR_VARIABLES__2,scores)
    # testing the model 2 on test set
    test_df2=test_df[__PREDICTOR_VARIABLES__2]
    print("Testing Model 2 against test data")
    predicted_class = clf2.predict(test_df2)    
    print("Done")
    print_test_results(2,actual_class,predicted_class)

    #  Model 3:
    # training model and performing 5 fold cross validation
    #  on predictor list 3
    clf3, scores = model(3,cleaned_df,__PREDICTOR_VARIABLES__3)    
    # printing cv scores for model with third set of predictor variables
    print_cv_results(3,__PREDICTOR_VARIABLES__3,scores)
    # testing the model 3 on test set
    test_df3=test_df[__PREDICTOR_VARIABLES__3]        
    print("Testing Model 3 against test data")
    predicted_class = clf3.predict(test_df3)    
    print("Done")
    print_test_results(3,actual_class,predicted_class)
    
    print("Done")
    print("#############################################################################################")
    print()
    return
### end of main


if __name__ == "__main__":    
    main()
    
