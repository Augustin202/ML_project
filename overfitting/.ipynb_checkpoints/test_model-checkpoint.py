import os
import sys
chemin=os.getcwd()
print(os.getcwd())
os.chdir('./..')
print(os.getcwd())
sys.path.append(os.getcwd())
from hack_class import *
os.chdir(chemin)
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import json

from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.datasets import fetch_openml

models_dic={'rf':RandomForestClassifier(), 'tree':DecisionTreeClassifier(), 'logistic':LogisticRegression(C=30.0),
      'knn':KNeighborsClassifier(), 'MLP':MLPClassifier()}

args = sys.argv[1:]
model_name = args[0]
num_fichier = int(args[1])

model=models_dic[model_name]
np.random.seed(42)

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
X = (X/255. - .5)*2

#Target model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=300, train_size=300,stratify = y)
rf_clf = clone(model)
rf_clf.fit(X_train, y_train)

random.seed()

reports=[]
for i in range(10):
    #on split les données en deux, on garde le set shadow qui va servir à entrainer les shadow models
    X_shadow, temp, y_shadow, temp2 = train_test_split(X, y, test_size=300, train_size=300,stratify = y)

    models = [clone(model)]*2
    test = HackingModel(RandomForestClassifier(n_estimators=100),models, X_shadow, y_shadow,
                      list(set(y_shadow)),list(set(y_train)) )
    report=test.print_score_hacking(X_train,y_train, X_test,y_test, rf_clf)
    reports.append(report)

# save
filename = "./data/report_test_model_{}_{}.json".format(model_name, num_fichier)
with open(filename, "w") as json_file:
    json.dump(reports, json_file, indent=2)
