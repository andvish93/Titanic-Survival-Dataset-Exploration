import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 


df = pd.read_csv("/Users/amerrez/CS286/Titanic/train.csv")
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

cols = df.columns

cm = np.corrcoef(df[['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values.T) 
sns.set(font_scale=1.5) 
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