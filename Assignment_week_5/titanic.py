import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("train.csv")

cols = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols, axis=1)

df = df.dropna()

dummies = []
cols = ['Pclass', 'Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(df[col]))

titanic_dummies = pd.concat(dummies, axis=1)

df = pd.concat((df,titanic_dummies), axis=1)

df['Age'] = df['Age'].interpolate()

X = df.values
y = df['Survived'].values

X = np.delete(X, 1, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)




