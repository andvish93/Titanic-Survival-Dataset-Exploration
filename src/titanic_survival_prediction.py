"""
CS 286 Data Analysis and Prediction Project

Predicting titanic survival



Authors: Saketh Saxena, Amer Rez, Anand V., Vedashree

Last updated:12/16/2018
"""

# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score

# __PREDICTOR_VARIABLES__ = ['Pclass','Age', 'SibSp', 'Parch', 'Fare', 'female', 'C', 'Q', 'S']
__PREDICTOR_VARIABLES__ = ['Fare', 'female']


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
    sns.barplot(x='Sex',y='Survived',data=df, palette= colorList)
    print("Percentage of females who survived:", df["Survived"][df["Sex"] == "female"].value_counts(normalize = True)[1]*100)
    
    print("Percentage of males who survived:", df["Survived"][df["Sex"] == "male"].value_counts(normalize = True)[1]*100)


    # Comparing the Pclass feature against Survived
    print("Plotting Pclass vs survived")
    plt.figure(4)
    
    sns.barplot(x='Pclass',y='Survived',data=df, palette= colorList)
    print("Percentage of Pclass = 1 who survived:", df["Survived"][df["Pclass"] == 1].value_counts(normalize = True)[1]*100)
    print("Percentage of Pclass = 2 who survived:", df["Survived"][df["Pclass"] == 2].value_counts(normalize = True)[1]*100)    
    print("Percentage of Pclass = 3 who survived:", df["Survived"][df["Pclass"] == 3].value_counts(normalize = True)[1]*100)



    print("Plotting Parch vs survived")
    plt.figure(5)
    sns.barplot(x='Parch',y='Survived',data=df, palette= colorList)
    # Comparing the Parch feature against Survived
    grouped_df = df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    print("\nPercentage of people who survived with following no of parents or children\n",grouped_df)



    # Comparing the Sibling  feature against Survived
    print("Plotting no of Siblings vs survived")
    plt.figure(6)
    sns.barplot(x='SibSp',y='Survived',data=df,palette= colorList)
    print("Percentage of SibSp = 0 who survived:", df["Survived"][df["SibSp"] == 0].value_counts(normalize = True)[1]*100)
    print("Percentage of SibSp = 1 who survived:", df["Survived"][df["SibSp"] == 1].value_counts(normalize = True)[1]*100)
    print("Percentage of SibSp = 2 who survived:", df["Survived"][df["SibSp"] == 2].value_counts(normalize = True)[1]*100)


    # Comparing the embarked  feature against Survived
    print("Plotting embarked vs survived")
    plt.figure(7)
    sns.barplot(x='Embarked',y='Survived',data=df,palette= colorList)
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
    sns.barplot(x="AgeGroup", y="Survived", data=df, palette= colorList)

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

    print("done")    
    return df
    
def model(df):
    
    clf = SVC(C = 1, gamma = 'auto', class_weight=None, coef0=0.0, kernel = 'linear')
    print("Training the model")
    clf.fit(df[__PREDICTOR_VARIABLES__], df["Survived"])

    print("Performing 5 Fold cross validation")

    scores = cross_validate(clf, df[__PREDICTOR_VARIABLES__], df["Survived"], cv=2,scoring=('accuracy', 'precision','recall','f1'),return_train_score=False)

    print("Done")

    return clf,scores

def main():
    
    # reading training file
    print("Reading training file")
    df = read_file("train.csv")
    
    # data visualization
    data_visualization(df)

    # data cleaning - pruning, imputation, one hot encoding
    cleaned_df = data_cleaning(df)

    # training and 5 fold and c parameter
    clf, scores = model(cleaned_df)    
    
    print("Cross validation Accuracy scores : ",scores["test_accuracy"])
    print("Cross validation Precision scores : ",scores["test_precision"])
    print("Cross validation Recall scores : ",scores["test_recall"])
    print("Cross validation f1 scores : ",scores["test_f1"])
    print("Score time: ",scores["score_time"])
    print("Score fit time: ",scores["fit_time"])

    
    test_df = read_file("test.csv")

    age_median = df["Age"].median()    
    Fare_median = df["Fare"].median()
    
    print("Imputing age column with the median value of training data")    
    test_df.Age.fillna(age_median,inplace=True)

    print("Imputing Fare column with the median value of test data")    
    
    test_df.Fare.fillna(Fare_median,inplace=True)

    # One hot encoding on feature Sex in test dataset
    test_df = one_hot_enc(test_df,"Sex")
    # One hot encoding on feature Embarked in test dataset
    test_df = one_hot_enc(test_df,"Embarked")
    
    test_df=test_df[__PREDICTOR_VARIABLES__]
        
    predicted_class = clf.predict(test_df)
    
    actual_class = pd.read_csv("gender_submission.csv")

    
    print("Accuracy % of predictions on test data",accuracy_score(actual_class["Survived"], predicted_class, normalize=True, sample_weight=None)*100)
    
    print("Precision of predictions on test data",precision_score(actual_class["Survived"], predicted_class,   average='macro'))
    
    print("Recall of predictions on test data",recall_score(actual_class["Survived"], predicted_class,  average='macro'))
  
    print("Confusion matrix :\n",confusion_matrix(actual_class["Survived"], predicted_class))
    
    return



if __name__ == "__main__":    
    main()
    