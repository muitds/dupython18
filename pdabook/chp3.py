##%% Data Manipulation with Pandas

#pandas requires numpys
import pandas
pandas.__version__

import pandas as pd

#%% Pandas objects

import numpy as np
import pandas as pd

#Pandas Series Object

data = pd.Series([0.25, 0.5, 0.75, 1])
data
# seq of values and indices
data.values
data.index
data[1]
data[1:3]

#Series
data = pd.Series([0.25, 0.5, 0.75, 1], index=['a','b','c','d'])
data

population_dict = {'delhi':1000, 'gurgaon':500, 'noida':800, 'faridabad':400}
population_dict
population_dict[1:2]  # error
population_dict['noida']  

#Series : pd.Series(data, index=index)
pd.Series([2,4,6])
pd.Series(5, index=[100,200,300])
pd.Series({ 2:'a', 1:'b',3:'c'})
pd.Series({ 2:'a', 1:'b',3:'c'}, index=[3,2])

# Data Frame
area_dict = {'delhi':40, 'gurgaon':30, 'noida':20, 'faridabad':15}
area= pd.Series(area_dict)
population = pd.Series(population_dict)

#states = pd.DataFrame(population_dict, area_dict)
population, area
states = pd.DataFrame({'population':population, 'area':area})
states
type(states)
states.values
states.index
states.columns

# DF as spl dictionary
states['area']

#Construct DF
pd.DataFrame(population)
pd.DataFrame(population, columns=['population'])

pd.DataFrame([population, area])

data
type(population)
pd.DataFrame([population, area])

# From List of Dictionaries
pd.DataFrame([ { 'a',1, 'b',2}, {'b':3,'c':4}])


# From Dictionary of Series
pd.DataFrame({'population':population, 'area':area})

# From 2 D NP array
import numpy as np
pd.DataFrame(np.random.rand(3,2), 
             columns=['foo','bar'], 
             index=['a','b','c'])

# From Structured Array
A = np.zeros(3, dtype=[('A','i8'),('B','f8')])
A
pd.DataFrame(A)

#Panda Index Object

ind = pd.Index([2,3,5,7,11])
ind

#Index immutable array
ind[1]
ind[::2]
ind.size, ind.shape, ind.ndim, ind.dtype
ind[1]=0  #error immutable

#index as ordered set
indA = pd.Index([1,3,5,7,9])
indB = pd.Index([2,3,5,7,11])
indA & indB # intersection
indA | indB # union
indA ^ indB # sym diff


#%% Pg 107 Indexing & Selection

#Data Selection in Series
#Series as dictionary
import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0])
data
data.index
data.index = ['a','b','c','d']
data
rollno = [109,102,105,106,103,110,101,107,104,111,108]
name = ['meena','apoorva','kastav','shubam', 'goldie',
        'hitesh', 'shruti','vijay','achal','lalit','varun']
gender =['F','F','M','M','M','M','F','M','M','M','M',]
python = np.random.randint(60,90,11)
sas = np.random.randint(65,85,11)

students = pd.Series(name, index=rollno)
students
112 in students  # rollno
111 in students
students.keys()
students.items
list(students.items())
students[111]='jain'
students

#Series as 1 D Array
students[0:5]
students[111:115]  # nothing

#indexers
students.loc[0]
students.iloc[0]
students.loc[111]
students.ix[111]
students.iloc[0:5]
students.loc[103:108]

# Data Selection in DF
area
population
states.area
states.population
states['area']
states['population']
states.area is states['area']

# DF as 2D Array
states.values

#%% Student Data
rollno = pd.Series([109,102,105,106,103,110,101,107,104,111,108])
name = pd.Series(['meena','apoorva','kastav','shubam', 'goldie',
        'hitesh', 'shruti','vijay','achal','lalit','varun'])
gender = pd.Series(['F','F','M','M','M','M','F','M','M','M','M'])
python = pd.Series(np.random.randint(60,90,11))
sas = pd.Series(np.random.randint(65,85,11))

studentDF = pd.concat([rollno, name, gender, python, sas], axis=1)
studentDF

studentDF2 = pd.DataFrame({'rollno':rollno, 'name':name, 'gender':gender,
                'python':python, 'sas':sas})
studentDF2
studentDF2.index = rollno
studentDF2

studentDF3 = pd.DataFrame({'rollno':rollno, 'sname':name, 'gender':gender,
                'python':python, 'sas':sas}, 
                 columns=['rollno','sname','gender', 'python', 'sas'])
studentDF3
studentDF3.index = rollno
studentDF3
studentDF3.values   # as array
studentDF3.T
studentDF3.values[0]  # first row
studentDF3['sname']
studentDF3.iloc[:3,:2]
studentDF3.loc[:105,:'python']
studentDF3.iloc[0:,0:2]
studentDF3.ix[:105,:'gender']  # can be confusion here
studentDF3.ix[:5,:'gender']  # error
studentDF3['total'] = studentDF3['python'] + studentDF3['sas']
studentDF3
studentDF3[studentDF3.total > 150]


#%% Operating on Pandas
#Ufuncs  work with Numpy and Pandas alike
# Index alignment

np.random.seed(0)
name1 = ['meena','apoorva','kastav','shubam', 'goldie',
        'hitesh', 'shruti','vijay','achal','lalit','varun']
python = pd.Series(np.random.randint(60,90,11), index=name1)

np.random.seed(10)
name2 = ['varun', 'meena','vijay', 'apoorva','lalit', 'kastav',
         'shubam', 'goldie','hitesh', 'shruti','achal']
sas = pd.Series(np.random.randint(65,85,11),index=name2)
python
sas
(python + sas)

np.random.seed(5)
name3 = [ 'meena','vijay', 'apoorva','kastav',
         'shubam', 'goldie','hitesh','achal']
hadoop = pd.Series(np.random.randint(65,85,8),index=name3)
hadoop
(python + sas + hadoop)
python + hadoop
python.add(hadoop, fill_value=0)
#will see how to add 3 series

# Index alignment in DF

A = pd.DataFrame(np.random.randint(0,20,(2,2)), columns=list('AB'))
B = pd.DataFrame(np.random.randint(0,10,(3,3)), columns=list('BAC'))
A + B

A = pd.DataFrame(python, index=name1)
B = pd.DataFrame(sas, index=name2)
C = pd.DataFrame(hadoop, index=name3)
pd.concat(python, sas, hadoop)
studentDF4 = pd.concat([A,B,C], axis=1)
studentDF4.columns = ['python','sas','hadoop']

studentDF4b =studentDF4.copy()
studentDF4b.fillna(0, inplace=True)
studentDF4b
studentDF4b



#%% Pg 118 Operations between DF and Series

x = np.random.randint(0,10,(3,4))
x
list('QRST')
df = pd.DataFrame(x, columns=['Q', 'R', 'S', 'T'])
df
df = pd.DataFrame(x, columns=list('QRST'))
df
type(df)
df.subtract(df['R'], axis=0)
df.subtract(df.iloc[0].values, axis=1)

#another way
first_row = df.iloc[[0]].values[0]
first_row
df.apply(lambda row: row - first_row, axis=1)
df
df.apply(lambda col: col - first_row, axis=1)  #row is just word
# axis tells it to opeate it row wise

# half row  - skip this
halfrow = df.iloc[0, ::2]
halfrow
df - halfrow

studentDF4.iloc[0,::2]


#%% Missing Data
#masking(Boolean_ or using sentinel values(999, NAN, NA)
#no built in notion of NA values
# python has got more data types than R

#using None
import numpy as np
import pandas as pd
vals = np.array([1, None, 3, 4])
vals
vals.sum()  # does not work
np.nansum(vals)  #not support

for dtype in ['object', 'int']:
    print("dtype =", dtype)
    %timeit np.arange(1E6, dtype=dtype).sum()
    print()
# int dtype takes less time
    

#NAN Missing numerical data
vals2 = np.array([1,np.nan, 3,4])
# special rep ; operations with nan will be nan like virus
1 + np.nan
vals2.sum()
np.nansum(vals2)
# many functions like this np.nanmin, np.nanmax

np.nansum(vals)

# Missing values in Pandas
NaN and None both are supported
pd.Series([1, np.nan, 2, None])

x = pd.Series(range(2), dtype=int)
x[0] = None
x  # converted to float upcast NaN - floating pt rep of missing values

#strings in Pandas is stored as object type
pd.Series(['dhiraj'])

#operating on Null Values
#isnull(), notnull(), dropna(), fillna()

#Detecting Null Values
data = pd.Series([1,np.nan, 'hello', None])
type(data)
data.dtype  # object
data.isnull()
data[data.notnull()]

#dropping null values
data.dropna()

studentDF4
studentDF4.dropna()
studentDF4.dropna(axis=1)
studentDF4.dropna(axis=0)
studentDF4.dropna(axis='columns', how='all')
# all are missing 
studentDF4['tableau'] = np.nan
studentDF4
studentDF4.dropna(axis='columns', how='all')  #tableau gone
studentDF4.dropna(axis='columns', thresh=10) #min nonna value
studentDF4

studentDF4.fillna(0)  # does not replace
studentDF4.fillna(method='ffill')  # does not replace
studentDF4.fillna(method='ffill', axis=1)  # does not replace


