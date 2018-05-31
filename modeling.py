# Dependencies

# Data Processing and Exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical Exploration Library
import scipy.stats as stats
from minepy import MINE

# Feature Engineering
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import preprocessing

# Modeling
from sklearn import linear_model
from sklearn import model_selection
from sklearn import grid_search
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor


# Self-Implemented
import preprocessing as pre


# Calculates the root-mean-square-error
def score(x,y):
	rmse = np.power(np.mean(np.square(x - y)), 0.5)
	return(rmse)

# Scoring function for grid search
# Grid search ranks parameters "higher" scores as better
def scorer(x, y):
    rank = 1/score(x,y)
    return(rank)

# Final evaluation of a model using root-mean-square-error
def score_final(model,i,l):
	rank = score(model.predict(i), l)
	return(rank)


# Get best parameters for model using grid search
def get_params(model, params, train_i, train_l):
    clf = grid_search.GridSearchCV(model,
                                  params,
                                  scoring = make_scorer(scorer),
                                  cv=5,
                                  iid=False)
    clf.fit(train_i, train_l)
    params = clf.best_params_
    
    return(params)