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
import pandas as pd
import numpy as np
import os
import random

import matplotlib.pyplot as plt
from collections import Counter
import sklearn
import sklearn.cluster
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import clone
from sklearn.exceptions import NotFittedError
from tqdm import tqdm
from sklearn.datasets import fetch_openml



args = sys.argv[1:]
nb_shadow = int(args[0])
num_fichier = int(args[1])

X=pd.read_csv('X_adult.csv')
y=pd.read_csv('y_adult.csv')['predic']

np.random.seed(42)

#Target model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5000, train_size=5000,stratify = y)
logistic_clf = LogisticRegression(C= 100)
logistic_clf.fit(X_train, y_train)

random.seed()

reports=[]
for i in range(10):
    X_shadow, temp, y_shadow, temp2 = train_test_split(X, y, test_size=5000, train_size=5000,stratify = y)
    models = [LogisticRegression(C=100)]*nb_shadow
    test = HackingModel(RandomForestClassifier(n_estimators=500),models, X_shadow, y_shadow,
                      list(set(y_shadow)),list(set(y_train)) )
    report=test.print_score_hacking(X_train,y_train, X_test,y_test, logistic_clf)
    reports.append(report)

# save
filename = "report_shadow_model_{}_{}.json".format(nb_shadow, num_fichier)
with open(filename, "w") as json_file:
    json.dump(list_lpy, json_file, indent=2)