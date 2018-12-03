import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print('*******Fetch train and test data*******')
DIR='/Users/vedashreebhandare/Documents/CS286/ProjectTitanicSurvival/TitanicData'
train_data = pd.read_csv(DIR+'/train.csv', delimiter=',')
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

#HeatMap using seborn
corr = train_data.corr()
_, ax = plt.subplots( figsize =( 12 , 10 ) )
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
_ = sns.heatmap(corr,cmap = cmap,square=True,cbar_kws={ 'shrink' : .9 },ax=ax,annot = True,annot_kws = { 'fontsize' : 12 })
plt.figure(1)

#Drop cabin since many missing values
df = train_data.drop(columns='Cabin')
print(df.head())


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
plt.figure(2)
sns.barplot(x='Sex',y='Survived',data=df, palette= colorList)
grouped_df = df.groupby('Sex',as_index=False).Survived.mean()
print("\n**********Percentage of Male and Female who survived**************\n",grouped_df)


#Comparing the Pclass feature against Survived
plt.figure(3)
sns.barplot(x='Pclass',y='Survived',data=df, palette= colorList)
grouped_df = df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print("\n**********Percentage of people in different class who survived**************\n",grouped_df)
#print(grouped_df)




#Comparing the Parch feature against Survived
plt.figure(4)
sns.barplot(x='Parch',y='Survived',data=df, palette= colorList)
grouped_df = df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print("\n**********Percentage of people who survived with following no of parents or children**************\n",grouped_df)
#print(grouped_df)



#Comparing the Sibling  feature against Survived
plt.figure(5)
sns.barplot(x='SibSp',y='Survived',data=df,palette= colorList)
grouped_df=df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print("**********Percentage of people who survived as per no of siblings**************\n",grouped_df)
#print(grouped_df)




#Survival  by Age 
#sort the ages into logical categories
df["Age"] = df["Age"].fillna(-0.5)
bins = [-1, 0, 5, 18, 35, 60, np.inf]
labels = ['Missing', 'Baby', 'Child', 'Teenager', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)
plt.figure(6)
#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=df, palette= colorList)
plt.show()




# scatter plot
# plt.scatter(df.Survived, df.Age)

# histogram: shows the conts
# df.hist(column='Sex', bins=25)

#
# plt.imshow(df.columns['Survived','Sex'], cmap='hot', interpolation='nearest')

## scatter plots for all pairs
# sns.set(style='whitegrid', context='notebook')
# ax = sns.pairplot(df[cols[1:3]],);
# # plt.savefig("scatter.pdf")

# plt.show()