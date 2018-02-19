# -*- coding: utf-8 -*-
"""
Mon Feb 19 09:45:40 2018: Dhiraj
"""

#Marks
import numpy as np
import pandas as pd

rng = np.random.RandomState(42)
marks = pd.Series(rng.randint(50,100,11))
marks

marks.sum()
marks.std()

# Dictionary
dict(x=1,y=4)
# Groupwise
df = pd.DataFrame({'A':rng.randint(1,10,6), 'B':rng.randint(1,10,6)})
df
df.mean()
df.mean(axis=0)
df.mean(axis=1)
df.mean(axis='columns')

df.describe()

# GroupBy
# Split - Apply - Combine
#Repeat
['A','B','C'] * 2
np.repeat(['A','B','C'] , 2)
np.repeat(['A','B','C'] , [1,2,3])

df = pd.DataFrame({'key':['A','B','C']* 2, 'data1':range(6), 'data2':rng.randint(0,10,6)}, columns=['key','data1','data2'] )

df

df.groupby('key')  # nothing will happen

df.groupby('key').aggregate(['min','max','median'])
df.groupby('key').aggregate([np.median,'median']) # error

df.groupby('key').aggregate({'data1':'min', 'data2':'max'})

#Filter  : Select columsn
df.filter(items = ['data1'])
df.filter(like = '2', axis=0)
df.groupby('key').std()

grouped = df.groupby('key')

grouped.filter(lambda x : x['data2'].mean() > 2)
grouped.filter(lambda x : x['data2'].std() > 2)
grouped.transform(lambda x: x - x.mean())


# Apply Method
df
grouped.apply(lambda x: x['data2'] * 2)


# Provide Group Keys
df.groupby('key').sum()
df.groupby(df['key']).sum()

# change the index to key values
df2 = df.set_index('key')
df2

newmap = {'A':'Post Graduate', 'B':'Master of Science','C':'Bachelor of Science'}
df2.groupby(newmap).sum()

df2.groupby(str.lower).mean()
#str  is key values

df2.groupby([str, str.lower, newmap]).mean()

# Stack
df.groupby('key').sum().unstack()
