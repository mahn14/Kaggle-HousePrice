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
from sklearn.preprocessing import Imputer




def get_df(file):
	df = pd.read_csv(file)

	# Log transform 'SalePrice'
	# Remove outliers
	if file == 'train.csv':
		df['SalePrice'] = np.log(df['SalePrice'])
		
		outliers = np.sort(df['SalePrice'])[-2:]
		df = df[df['SalePrice'] < np.min(outliers)].reset_index(drop=True)

	return(df)



def remove_missing(file):
	df = get_df(file)

	train = True
	test = False
	if 'SalePrice' not in list(df.columns):
		test = True
		train = False

	# Drop features missing more than 10% of rows
	drop = df.isnull().sum()/df.shape[0]
	drop_list = list(drop[drop > 0.1].index)
	df = df.drop(drop_list, axis=1)

	# Separate remaining features with missing rows from df
	keep = df.isnull().sum() / df.shape[0]
	keep_list = keep[keep > 0].index 
	df_missing = df[keep_list]
	df = df.drop(keep_list, axis=1)

	# Label continuous (num) columns
	if train:
		num = ['GarageYrBlt', 'MasVnrArea']
	else:
		num = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageArea', 'GarageCars', 'GarageYrBlt', 'TotalBsmtSF', 'MasVnrArea']

	# Label categorical (cat) columns
	cat = []
	for i in  df_missing.columns:
	    if i not in num:
	        cat.append(i)

	# Separate cont and cat 
	df_missing_num = df_missing[num]
	df_missing_cat = df_missing[cat]

	# TESTING THIS OUT
	df_missing_cat['MasVnrType'] = df_missing_cat['MasVnrType'].fillna('None')
	for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
		df_missing_cat[col] = df_missing_cat[col].fillna('None')
	df_missing_cat['Electrical'] = df_missing_cat['Eletrical'].fillna(df_missing_cat['Electrical'].mode()[0])
	for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
		df_missing_cat[col] = df_missing_cat[col].fillna('None')







	'''
	if train:		
		df_missing_num['MasVnrArea']= df_missing_num['MasVnrArea'].fillna(0)
	
	# Impute using mean/'None'
	df_missing_cat = df_missing_cat.fillna('None')
	imputer = Imputer(missing_values='NaN', strategy='median')
	X = imputer.fit_transform(df_missing_num)
	df_missing_num = pd.DataFrame(X, columns=num)
	'''

	# Concatenate dfs
	df_missing = pd.concat([df_missing_num, df_missing_cat], axis=1)
	df = pd.concat([df,df_missing],axis=1)

	return(df)




def get_data(file):
	pre = remove_missing(file)
	if file == 'train.csv':
		salePrice = pre['SalePrice']
		train = pre.drop('SalePrice',axis=1)

	# Get continuous (num) features
	all_num = train.dtypes[train.dtypes != 'object'].index
	df_num = train[all_num]

	# Extract principal component from highly correlated features
	pca = decomposition.PCA(1)
	X1 = pca.fit_transform(df_num[['TotalBsmtSF', '1stFlrSF']])
	X1 = pd.DataFrame(X1)
	X1.columns = ['TotalBsmtSF_1stFlrSF']

	pca = decomposition.PCA(1)
	X2 = pca.fit_transform(df_num[['GrLivArea', 'TotRmsAbvGrd']])
	X2 = pd.DataFrame(X2)
	X2.columns = ['GrLivArea_TotRmsAbvGrd']

	pca = decomposition.PCA(1)
	X3 = pca.fit_transform(df_num[['GarageArea', 'GarageCars']])
	X3 = pd.DataFrame(X3)
	X3.columns = ['GarageArea_GarageCars']

	# Concatenate df post-PCA
	df_num = df_num.drop(['TotalBsmtSF', '1stFlrSF','GrLivArea', 'TotRmsAbvGrd','GarageArea', 'GarageCars'], axis=1)
	df_num = pd.concat([df_num, X1, X2, X3], axis=1)

	# Separate continuous (num) into discrete (cat)
	# 	using number of unique 
	value_count = [(len(df_num[i].value_counts()), i) for i in df_num.columns]
	num_to_cat = []
	for i in range(len(value_count)):
	    if value_count[i][0] < 60:
	        num_to_cat.append(value_count[i][1])
	df_num_to_cat = df_num[num_to_cat]
	df_num = df_num.drop(num_to_cat, axis=1)

	# Scaling and Normalizing continuous (num) features
	scaler = preprocessing.MinMaxScaler()
	df_num = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns)

	from scipy.special import boxcox1p
	for i in df_num:
	    df_num[i] += (df_num[i].describe()['min']+1)
	    
	    if stats.skew(df_num[i]) < 0:
	        df_num[i] = boxcox1p(df_num[i], 2)
	    else:
	        df_num[i] = boxcox1p(df_num[i], -0.1)   

	# Prepare discrete (cat) features
	all_cat = train.dtypes[train.dtypes == 'object'].index
	df_cat = train[all_cat]
	df_cat = pd.concat([df_cat, df_num_to_cat], axis=1)
	dummies = pd.get_dummies(df_cat)

	# Final df
	final = pd.concat([df_num, dummies], axis=1)
	final = final.drop('Id', axis=1)
	if file == 'train.csv':
		final = pd.concat([final, salePrice],axis=1)
	return(final)



def split_train_test(df, frac=0.25):

	df = df.sample(frac = 1)
	split_index = int(np.round(df.shape[0] * frac))

	train = df.iloc[split_index:, :]
	test = df.iloc[:split_index, :]

	train_l = train['SalePrice']
	train_i = train.drop('SalePrice', axis=1)
	test_l = test['SalePrice']
	test_i = test.drop('SalePrice', axis=1)
	

	return(train_i, train_l, test_i, test_l)





