import pandas as pd
import numpy as np
import os

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


#This class describes only one shadow model
class  ShadowModel:
    #takes a sklearn model, and datasets for training and testing
    #list_y_class is the list of values that y can take
    def __init__(self, model, X_train, X_test, y_train, y_test, liste_y_class):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.liste_y_class = liste_y_class
        self.model_fit(X_train, y_train)
        self.model_pred_proba()

    #fit self.model according to X and y
    def model_fit(self,X,y):
        self.X_train = X
        self.y_train = y
        self.model.fit(X, y)

    #this function creates the database we will use for the hack model
    #to every line of X, we have to add the prediction probabilities, and if wether or not it was in the training set
    #this cell is a bit technical, because all the elements available in y might not match all the possible values of y
    def model_pred_proba(self):
        #results for the train set
        predictions_df_temp = pd.DataFrame(self.model.predict_proba(self.X_train), columns=list(set(self.y_train))).reset_index(drop=True)
        predictions_df = pd.DataFrame([[0]*len(self.liste_y_class) for i in range(len(predictions_df_temp))], columns=self.liste_y_class)
        predictions_df[predictions_df_temp.columns]= predictions_df_temp

        X_train_proba = pd.concat([self.X_train.reset_index(drop=True), predictions_df.reset_index(drop=True)],  axis=1)
        X_train_proba['predic'] = self.y_train.reset_index(drop=True)
        X_train_proba['entrainement'] = 1 #indicates that these elements where in the train set

        #results for the test set
        predictions_df_temp2 = pd.DataFrame(self.model.predict_proba(self.X_test), columns=list(set(self.y_train))).reset_index(drop=True)
        predictions_df2 = pd.DataFrame([[0]*len(self.liste_y_class) for i in range(len(predictions_df_temp2))], columns=self.liste_y_class)
        predictions_df2[predictions_df_temp2.columns]= predictions_df_temp2
        X_test_proba = pd.concat([self.X_test.reset_index(drop=True), predictions_df2.reset_index(drop=True)],  axis=1)
        X_test_proba['predic'] = self.y_test.reset_index(drop=True)
        X_test_proba['entrainement'] = 0 #indicates that these elements where in the test set

        self.X_proba = pd.concat([X_train_proba,X_test_proba])

        
#this class creates all the shadow models we are going to use. Hence it contains multiple shadow models.
class ShadowModels:
    #list_y_class contains all the values that y can take
    def __init__(self, models, X, y,list_y_class):
        self.X = X
        self.y = y
        self.list_y_class = list_y_class
        self.fit_models(models, list_y_class)
        self.concatenate_data()

    #this functions create a shadow model for every element of the list models
    #modelsis a list of sklearn models. Dataset are equally split into dataset and training set
    def fit_models(self, models,list_y_class):
        n = len(self.X)//2
        self.shadow_models = []
        for model in models:
            X_train, X_test, y_train, y_test=train_test_split(self.X, self.y, test_size = n, train_size= n,stratify = self.y)
            self.shadow_models.append(ShadowModel(model, X_train, X_test, y_train, y_test,list_y_class))

    #this function concatenates all the databases created by the shadow models, in order to create a dataframe for the hacking model
    def concatenate_data(self):
        data_attaquant= self.shadow_models[0].X_proba
        for shadow_model in self.shadow_models[1:]:
            data_attaquant = pd.concat([data_attaquant, shadow_model.X_proba],ignore_index=True ,axis=0)
        self.data_attaquant = data_attaquant
        
        
#this class trains the hacking model
class HackingModel:
    #entries : hack_model (sklearn model, the one we will train and will perform hacking), shadow_models (list of sklearn models that we will use as shadow models),
    #list_y_class (all the values y can take), list_y_target_model (all the y values that were in the dataset used by the target model)
    def __init__(self, hack_model, shadow_models, X, y, list_y_class,list_y_target_model):
        self.X = X
        self.y = y
        self.list_y_class = list_y_class
        self.list_y_target_model = list_y_target_model
        #creation of an object ShadowModels that contains all the ShadowModel we will use for hacking
        self.shadowmodels = ShadowModels(shadow_models, X, y, list_y_class)
        self.hack_models = {}
        #we create a hack_model for each class of y
        for class_y in self.list_y_class:
            self.hack_models[class_y] = clone(hack_model)

        self.set_data_per_class()
        self.fit_hack_models()

    #creates a list of datasets, corresponding to each value that y can take
    def set_data_per_class(self):
        self.list_entrainement = []
        for class_y in self.list_y_class:
            y = self.shadowmodels.data_attaquant[self.shadowmodels.data_attaquant['predic']==class_y].entrainement
            X = self.shadowmodels.data_attaquant[self.shadowmodels.data_attaquant['predic']==class_y].drop(columns=['entrainement'])
            X.columns = X.columns.astype(str)
            self.list_entrainement.append([X,y])

    #this function fits all the hack models (that is to say for every value of y)
    def fit_hack_models(self):
        for (datas, class_y) in tqdm(zip(self.list_entrainement, self.list_y_class)):
            if len(datas[0])>0:
                self.hack_models[class_y].fit(datas[0], datas[1])

    #this function predicts for every x in X if the model was trained on it or not.
    #For every value of y, we use the appropriated hack model
    def predict(self, X):
        predictions = []
        for class_y in self.list_y_class:
            X_class_y = X[X['predic']==class_y]
            index = X_class_y.index
            if len(X_class_y)>0:
                #predictions.append(pd.DataFrame(self.hack_models[class_y].predict(X_class_y), index=index))
                try:
                    predictions.append(pd.DataFrame(self.hack_models[class_y].predict(X_class_y), index=index))
                except NotFittedError as e:
                    predictions.append(pd.DataFrame([0]*len(X_class_y), index=index))

        return pd.concat(predictions,axis=0)

    #this function, given X and y, predicts which elements were in the dataset or not.
    #target model is the model we want to hack
    def predict_target_model(self, X, y, target_model):
        prob = target_model.predict_proba(X)
        prob = pd.DataFrame(prob, columns=self.list_y_target_model).reset_index(drop=True)
        prob_p = pd.DataFrame([[0]*len(self.list_y_class) for i in range(len(prob))], columns=self.list_y_class)
        prob_p[prob.columns] = prob

        X = pd.concat([X.reset_index(drop=True), prob_p.reset_index(drop=True)],  axis=1)
        X['predic'] = y.reset_index(drop=True)
        X.columns = X.columns.astype(str)
        return self.predict(X)


    #this function gives the classification report of the model
    #X_in and y_in were in the training dataset of the target model, X_out and y_out were not
    def print_score_hacking(self, X_in, y_in, X_out, y_out, target_model):
        res_in = self.predict_target_model(X_in, y_in, target_model)[0]
        res_out = self.predict_target_model(X_out, y_out, target_model)[0]
        res = np.concatenate((res_in.values,res_out.values))
        true_values = [1]*len(res_in) + [0]*len(res_out)
        report = classification_report(true_values, res)
        print("Classification Report, test set:")
        print(report)
        return classification_report(true_values, res,output_dict = True)

    def give_constants(self, X_in, y_in, X_out, y_out, target_model):
        res_in = self.predict_target_model(X_in, y_in, target_model)[0]
        res_out = self.predict_target_model(X_out, y_out, target_model)[0]
        res = np.concatenate((res_in.values,res_out.values))
        true_values = [1]*len(res_in) + [0]*len(res_out)

        report = classification_report(true_values, res,output_dict = True)
        return report['accuracy'],report['weighted avg']['f1-score']
