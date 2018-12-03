import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 


df = pd.read_csv("/Users/amerrez/CS286/Titanic/train_split.csv")
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

cols = df.columns
selectedColumns = ['Survived', 'Age', 'SibSp', 'Parch', 'Fare','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S','Pclass_1','Pclass_2','Pclass_3']

cm = np.corrcoef(df[['Survived', 'Age', 'SibSp', 'Parch', 'Fare','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S','Pclass_1','Pclass_2','Pclass_3']].values.T) 
sns.set(font_scale=1.5)

threshold = 0.8
## print the features corelated with the target :
print("The features corelated with the target based on threshold " + str(threshold))
for rowIndex in range(len(cm)):
    corrIndex = 0 # the target index
    if rowIndex != corrIndex:
        if (cm[rowIndex][corrIndex] > threshold or cm[rowIndex][corrIndex] < -threshold) :
            print(selectedColumns[rowIndex] + " and " + selectedColumns[corrIndex] + " are dependent")
print("\n")

## print the features corelated with each others
print("The features corelated with each others based on threshold " + str(threshold))
for rowIndex in range(1, len(cm)):
    for corrIndex in range(1, len(cm[rowIndex])):
        if rowIndex != corrIndex:
            if (cm[rowIndex][corrIndex] > threshold or cm[rowIndex][corrIndex] < -threshold):
                print(selectedColumns[rowIndex] + " and " + selectedColumns[corrIndex] + " are dependent")
            
## plot the Heatmap 
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols) 
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