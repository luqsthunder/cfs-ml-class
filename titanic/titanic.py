import pandas as pd
import math

train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')
all = [train_df, test_df]

print(train_df.columns.values)
important_features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
                      'Fare', 'Cabin', 'Embarked']
non_numerical_features = ['Sex', 'Pclass', 'Embarked']


# %% converting non-numerical-features

train_df['Sex'] = train_df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
train_df['Embarked'] = \
train_df['Embarked'].apply(lambda x: x if str(x) == 'nan' else {'S': 0,
                                                                'C': 1,
                                                                'Q': 2}[x])

def cabin_to_float(cabin_str):
    res = 0
    for cabin_str_split in cabin_str.split(' '):
        if len(cabin_str_split) > 1:
            res += ord(cabin_str_split[0:1]) * int(cabin_str_split[1:])
        else:
            res += ord(cabin_str_split[0:1])

    return res

train_df['Cabin'] = train_df['Cabin'].apply(lambda x: x if str(x) == 'nan' else
                                            cabin_to_float(x))

# %% achando quais features tem nan e preencher elas com a mediana
for i in important_features:
    print("feat: {}, unique: {}".format(i, train_df[i].isnull().unique()))

qfeatures_with_nan = ['Age', 'Cabin', 'Embarked']

import numpy as np

age_histo = np.histogram(train_df['Age'].dropna(),
                         bins = len(train_df['Age'].dropna().unique()))
cabin_histo = np.histogram(train_df['Cabin'].dropna(),
                           bins = len(train_df['Cabin'].dropna().unique()))
embarked_histo = np.histogram(train_df['Embarked'].dropna(),
                              bins = len(train_df['Embarked'].dropna()
                                                             .unique()))
