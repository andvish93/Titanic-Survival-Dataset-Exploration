import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import Imputer

print('*******Fetch train and test data*******')

train_data = pd.read_csv('train_split.csv', delimiter=',')
print(train_data.shape)

print('**********Train Information**********')
train_data.info()

print('**********Train Description**********')
print(train_data.describe(include='all'))


#Finding the percantage of missing values in train dataset
train_data.isnull().sum()/ len(train_data) *100

print('*******Columns with Null values*********')
print(train_data.isnull().sum(axis = 0))

print('*******Replace Null with *********')
train_data.replace(' ', np.nan)
print(train_data.shape)

print('*******Remove Null*********')
train_data = train_data.dropna()
print(train_data.shape)
train_data.info()


cols = train_data.columns


#Drop cabin since many missing values
df = train_data.drop(columns='Cabin')
print(df.head())



column_name = ['Survived', 'Sex_female', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
df_corr = df[column_name]

#HeatMap using seborn
plt.figure(1)
sns.heatmap(df_corr.corr(), 
            annot=True,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            vmax = .9,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20);




# scatter plot matrix
#plt.figure(2)
pd.plotting.scatter_matrix(df,figsize=(10,10))


#Number of Males or Female in the population
# plt.figure(2)
# sns.countplot('Sex',data=df)
# count = df['Sex'].value_counts()
# print("**************Count of Males and Females**********\n",count)

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

#Percentage of Male and Female who survived
plt.figure(3)
print("---------Sex Feature Analysis------")
sns.barplot(x='Sex_female',y='Survived',data=df, palette= colorList)
print("Percentage of females who survived:", df["Survived"][df["Sex_female"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", df["Survived"][df["Sex_female"] == 0].value_counts(normalize = True)[1]*100)


#Comparing the Pclass feature against Survived
plt.figure(4)
sns.barplot(x='Pclass',y='Survived',data=df, palette= colorList)
print("Percentage of Pclass = 1 who survived:", df["Survived"][df["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", df["Survived"][df["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", df["Survived"][df["Pclass"] == 3].value_counts(normalize = True)[1]*100)




#Comparing the Parch feature against Survived
plt.figure(5)
sns.barplot(x='Parch',y='Survived',data=df, palette= colorList)
grouped_df = df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print("\n**********Percentage of people who survived with following no of parents or children**************\n",grouped_df)
#print(grouped_df)



#Comparing the Sibling  feature against Survived
plt.figure(6)
sns.barplot(x='SibSp',y='Survived',data=df,palette= colorList)
print("Percentage of SibSp = 0 who survived:", df["Survived"][df["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", df["Survived"][df["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", df["Survived"][df["SibSp"] == 2].value_counts(normalize = True)[1]*100)



#Survival  by Age 
#sort the ages into logical categories

df["Age"] = df["Age"].fillna(-0.5)
bins = [0, 5, 18, 35, 60, np.inf]
labels = ['Baby', 'Child', 'Teenager', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)
plt.figure(7)
#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=df, palette= colorList)
plt.show()



#imputaion of mean to age column 
imp=Imputer(missing_values="NaN", strategy="mean" )
imp.fit(df[["Age"]])
df["Age"]=imp.transform(df[["Age"]]).ravel()
