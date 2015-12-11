# this is a mediocre script

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import numpy as np 

h = 'hello!'
print h
print "is it me..."
print "you're looking for?"
print "...I'm out of lyrics"

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
df['species'] = pd.Categorical(iris.target, iris.target_names)
df.head()

train, test = df[df['is_train']==True], df[df['is_train']==False]

features = df.columns[:4]
clf = RandomForestClassifier(n_jobs=2)
y, _ = pd.factorize(train['species'])
clf.fit(train[features], y)

predictions = iris.target_names[clf.predict(test[features])]
table = pd.crosstab(test['species'], predictions, rownames=['actual'], colnames=['preds'])
print table