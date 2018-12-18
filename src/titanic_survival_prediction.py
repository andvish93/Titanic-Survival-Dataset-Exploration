"""
CS 286 Data Analysis and Prediction Project

Predicting titanic survival



Authors: Saketh Saxena, Amer Rez, Anand V., Vedashree

Last updated:12/16/2018
"""

# imports
try:
    import numpy as np
except:
    print("Please install numpy ")
try:
    import pandas as pd
except:
    print("Please install pandas")
try:
    import matplotlib.pyplot as plt
ex
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score

__PREDICTOR_VARIABLES__1 = [1,2,3,'Age', 'SibSp', 'Parch', 'Fare', 'female', 'C', 'Q', 'S']
__PREDICTOR_VARIABLES__2 = ['Fare', 'female']
__PREDICTOR_VARIABLES__3 = [3, 'female']


# Read in a file, returns pandas dataframe
def read_file(filename):
    try:
        print("Reading the data set named: ", filename)
        df = pd.read_csv(filename)
    except IOError:
        raise IOError("Problems locating or opening the dataset named " + filename)
    print("Completed reading ", filename)
    print()
    return df

def one_hot_enc(df, column_name):
    one_hot = pd.get_dummies(df[column_name])
    df = df.drop(column_name,axis = 1)
    df = df.join(one_hot)
    return df
    
def print_cv_results(__PREDICTOR_VARIABLES__,scores):
    # printing cross validation scores    
    print("Cross validation scores for model with predictors",__PREDICTOR_VARIABLES__)
    print("Cross validation Accuracy scores : ",scores["test_accuracy"])
    print("Mean Cross validation accuracy :",scores["test_accuracy"].mean())
    print("Standard deviation of Cross validation accuracy :",scores["test_accuracy"].std()*2.0)
    print("Cross validation Precision scores : ",scores["test_precision"])
    print("Cross validation Recall scores : ",scores["test_recall"])
    print("Cross validation f1 scores : ",scores["test_f1"])
    print("Score time: ",scores["score_time"])
    print("Score fit time: ",scores["fit_time"])     
    return

def print_test_results(model_no, actual_class,predicted_class):
    print("Results for model no: ",model_no)
    # printing model performance scores based on test data
    print("Accuracy % of predictions on test data",accuracy_score(actual_class["Survived"], predicted_class, normalize=True, sample_weight=None)*100)    
    print("Precision of predictions on test data",precision_score(actual_class["Survived"], predicted_class,   average='macro'))    
    print("Recall of predictions on test data",recall_score(actual_class["Survived"], predicted_class,  average='macro'))  
    print("Confusion matrix :\n",confusion_matrix(actual_class["Survived"], predicted_class))
    return


def feature_correlation_analysis(df,threshold = 0.2):

    # One hot encoding on feature Sex
    print("Performing one hot encoding on Sex column")
    df = one_hot_enc(df,"Sex")
    
    # One hot encoding on feature Embarked
    print("Performing one hot encoding on Embarked column")
    df = one_hot_enc(df,"Embarked")
    
    # One hot encoding on feature Pclass
    print("Performing one hot encoding on Pclass column")
    df = one_hot_enc(df,"Pclass")    
    
    selectedColumns = list(df.columns)
    selectedColumns.remove("PassengerId")
    selectedColumns.remove("Cabin")
    selectedColumns.remove("Name")
    selectedColumns.remove("Ticket")
    

    
    cm = np.corrcoef(df[selectedColumns].values.T) 
    

    ## print the features corelated with the target
    print("The features corelated with the target based on threshold " + str(threshold))
    for rowIndex in range(len(cm)):
        corrIndex = 0 # the target index
        if rowIndex != corrIndex:
            if (cm[rowIndex][corrIndex] > threshold or cm[rowIndex][corrIndex] < -threshold) :
                print(str(selectedColumns[rowIndex]) + " and " + str(selectedColumns[corrIndex]) + " are dependent")
    print("\n")
    
    ## print the features corelated with each others
    print("The features corelated with each others based on threshold " + str(threshold))
    for rowIndex in range(1, len(cm)):
        for corrIndex in range(1, len(cm[rowIndex])):
            if rowIndex != corrIndex:
                if (cm[rowIndex][corrIndex] > threshold or cm[rowIndex][corrIndex] < -threshold):
                    print(str(selectedColumns[rowIndex]) + " and " + str(selectedColumns[corrIndex]) + " are dependent")
    print(selectedColumns)
    return

                        
def data_visualization(df):        
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
                        
    print("Plotting sex vs survived")
    # Percentage of Male and Female who survived
    plt.figure(3)
    print("Sex Feature Analysis")
    sns.barplot(x='Sex',y='Survived',data=df, palette= colorList).set_title("Survival Rate Vs Sex")
    print("Percentage of females who survived:", df["Survived"][df["Sex"] == "female"].value_counts(normalize = True)[1]*100)
    
    print("Percentage of males who survived:", df["Survived"][df["Sex"] == "male"].value_counts(normalize = True)[1]*100)


    # Comparing the Pclass feature against Survived
    print("Plotting Pclass vs survived")
    plt.figure(4)
    
    sns.barplot(x='Pclass',y='Survived',data=df, palette= colorList).set_title("Survival Rate Vs Class")
    print("Percentage of Pclass = 1 who survived:", df["Survived"][df["Pclass"] == 1].value_counts(normalize = True)[1]*100)
    print("Percentage of Pclass = 2 who survived:", df["Survived"][df["Pclass"] == 2].value_counts(normalize = True)[1]*100)    
    print("Percentage of Pclass = 3 who survived:", df["Survived"][df["Pclass"] == 3].value_counts(normalize = True)[1]*100)



    print("Plotting Parch vs survived")
    plt.figure(5)
    sns.barplot(x='Parch',y='Survived',data=df, palette= colorList).set_title("Survival Rate Vs Parch")
    # Comparing the Parch feature against Survived
    grouped_df = df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    print("\nPercentage of people who survived with following no of parents or children\n",grouped_df)



    # Comparing the Sibling  feature against Survived
    print("Plotting no of Siblings vs survived")
    plt.figure(6)
    sns.barplot(x='SibSp',y='Survived',data=df,palette= colorList).set_title("Survival Rate Vs SibSp")
    print("Percentage of SibSp = 0 who survived:", df["Survived"][df["SibSp"] == 0].value_counts(normalize = True)[1]*100)
    print("Percentage of SibSp = 1 who survived:", df["Survived"][df["SibSp"] == 1].value_counts(normalize = True)[1]*100)
    print("Percentage of SibSp = 2 who survived:", df["Survived"][df["SibSp"] == 2].value_counts(normalize = True)[1]*100)


    # Comparing the embarked  feature against Survived
    print("Plotting embarked vs survived")
    plt.figure(7)
    sns.barplot(x='Embarked',y='Survived',data=df,palette= colorList).set_title("Survival Rate Vs Embarked")
    print("Percentage of SibSp = 0 who survived:", df["Survived"][df["Embarked"] == "S"].value_counts(normalize = True)[1]*100)
    print("Percentage of SibSp = 1 who survived:", df["Survived"][df["Embarked"] == "Q"].value_counts(normalize = True)[1]*100)
    print("Percentage of SibSp = 2 who survived:", df["Survived"][df["Embarked"] == "C"].value_counts(normalize = True)[1]*100)

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

    plt.show()
    print("Done")
    
    return

def data_cleaning(df):
    
    # Drop cabin since many missing values
    print("Dropping column Cabin due to missing values")
    df = df.drop(columns='Cabin')
    print(df.head())
    
    print("Dropping rows with empty values for Embarked")
    df = df[pd.notnull(df['Embarked'])]

    # Imputing age column with mean values for the column    
    print("Imputing age column with the mean value")
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
    
    print("done")    
    return df
    
def model(df,__PREDICTOR_VARIABLES__):
    
    clf = SVC(C = 1, gamma = 'auto', class_weight=None, coef0=0.0, kernel = 'linear')
    print("Training the model")
    clf.fit(df[__PREDICTOR_VARIABLES__], df["Survived"])

    print("Performing 5 Fold cross validation")

    scores = cross_validate(clf, df[__PREDICTOR_VARIABLES__], df["Survived"], cv=5,scoring=('accuracy', 'precision','recall','f1'),return_train_score=False)

    print("Done")

    return clf,scores



def main():
    
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
    
    # extracting the median age and fare values of the training set
    age_median = df["Age"].median()    
    Fare_median = df["Fare"].median()    

    # reading in test file
    test_df = read_file("test.csv")
    
    # reading in actual classes
    actual_class = pd.read_csv("gender_submission.csv")
    
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


    # training and 5 fold and c parameter
    clf1, scores = model(cleaned_df,__PREDICTOR_VARIABLES__1)    
    # printing cv scores for model with first set of predictor variables
    print_cv_results(__PREDICTOR_VARIABLES__1,scores)    
    test_df1=test_df[__PREDICTOR_VARIABLES__1]        
    
    # testing the model 1 on test set
    predicted_class = clf1.predict(test_df1)    
    print_test_results(1,actual_class,predicted_class)
    

    # training and 5 fold and c parameter
    clf2, scores = model(cleaned_df,__PREDICTOR_VARIABLES__2)        
    # printing cv scores for model with first set of predictor variables
    print_cv_results(__PREDICTOR_VARIABLES__2,scores)
    
    test_df2=test_df[__PREDICTOR_VARIABLES__2]        
    # testing the model 2 on test set
    predicted_class = clf2.predict(test_df2)    
    print_test_results(2,actual_class,predicted_class)

    # training and 5 fold and c parameter for model 3
    clf3, scores = model(cleaned_df,__PREDICTOR_VARIABLES__3)    
    # printing cv scores for model with first set of predictor variables
    print_cv_results(__PREDICTOR_VARIABLES__3,scores)

    test_df3=test_df[__PREDICTOR_VARIABLES__3]        
    # testing the model 3 on test set
    predicted_class = clf3.predict(test_df3)    
    print_test_results(3,actual_class,predicted_class)

    return



if __name__ == "__main__":    
    main()
    
