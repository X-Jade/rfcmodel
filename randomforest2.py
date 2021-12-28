import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import json
from sklearn.tree import _tree
import dump_tree
import arff

# dataset = pd.read_csv("phishingdataset.csv")
# dataset = dataset.drop(
#     ["id", "Domain_registeration_length", "Abnormal_URL", "Redirect", "on_mouseover", "RightClick",
#     "popUpWidnow", "age_of_domain", "DNSRecord", "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page", "Statistical_report"],
#     1)

dataset = arff.load(open('dataset.arff', 'r'))
data = np.array(dataset["data"])
# dataset = [[-1,1,-1],
#             [-1,0,1]]

# data = { 'no':['1', '2', '3', '4', '5', '6'], 'idx':['a','b','c','d','e','f'],'country':['Canada', 'Portugal', 'Ireland', 'Nigeria', 'Brazil', 'India'] ,'continent':['America','Europe','Europe','Africa','SA','Asia'] }
# df = pandas.DataFrame(data, columns = ['nomor', 'index','country', 'continent'])

data = data[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 22, 30]]

# x = dataset.iloc[ : , :-1].values
# # y = dataset.iloc[ : ,-1: ].values
# y = dataset.iloc[ : ,-1:].values
X, y = data[:, :-1], data[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=40)

# params = [{'n_estimators':[100,300,500,700], 'max_features':['sqrt','log2'], 'criterion':['gini', 'entropy']}]
params = {'n_estimators':[100], 'max_features':['sqrt'], 'criterion':['entropy']}

# cv = cross validation
grid_search = GridSearchCV(RandomForestClassifier(), params, cv = 10, n_jobs=-1) 
grid_search.fit(x_train, y_train)

print("Best accuracy = " +str(grid_search.best_score_))
print("Best params = " +str(grid_search.best_params_))
# rfc = RandomForestClassifier(**grid_search.best_params_)
rfc = RandomForestClassifier()
# rfc = RandomForestClassifier(n_estimators=500, criterion='gini', max_features='log2')
rfc.fit(x_train, y_train)

y_predict = rfc.predict(x_test)

# joblib.dump(rfc, 'randomforest.txt')

df = pd.DataFrame(grid_search.cv_results_)
df = df.sort_values("rank_test_score")
df.to_csv("cv_results.csv")

test_data = dict()
test_data['X_test'] = x_test.tolist()
test_data['y_test'] = y_test.tolist()
with open('./testdata.json', 'w') as tdfile:
    json.dump(test_data, tdfile)

def forest_to_json(rfc):
    forest_json = dict()
    forest_json['n_features'] = rfc.n_features_
    forest_json['n_classes'] = rfc.n_classes_
    forest_json['clasess'] = rfc.classes_.tolist()
    forest_json['n_outputs'] = rfc.n_outputs_
    forest_json['n_estimators'] = rfc.n_estimators
    forest_json['estimators'] = [dump_tree.tree_to_json(estimator) for estimator in rfc.estimators_]
    return forest_json

json.dump(forest_to_json(rfc), open('./classifier.json', 'w'))
# print(type(dataset))
# print(x)
# print("=======")
# print(y)


# https://www.educba.com/pandas-dataframe-iloc/
# https://machinelearningknowledge.ai/python-sklearn-random-forest-classifier-tutorial-with-example/
