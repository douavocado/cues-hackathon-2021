#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 11:44:41 2021

@author: jx283
"""
import pandas as pd
import numpy as np

target = 'tower_height'

db = pd.read_excel('edited3.xlsx')

db = db.rename(columns={'Type':'type','Region':'region','Turbine rating (kW)': 'turbine_rating','Metocean': 'metocean','Blade length (m)':'blade_length','Operator': 'operator',"Water depth ": "water_depth",'Project Ref No.':'project_ref', "Tower height (m)": "tower_height", "Single Blade Weight (te)": "single_blade_weight", 'Built duration': 'built_duration', 'Nacelle Weights': 'nacelle_weights'})
#print(dir(db))
cols = list(db.columns)
db[cols] = db[cols].replace({'0':np.nan, 0:np.nan})
db['water_depth'] =  db['water_depth'].replace({'Deep':3, 'Mid':2, 'Shallow':1})
# artbitrary constant
db['constant'] = np.ones((len(db),1))
db['metocean'] =  db['metocean'].replace({'Moderate':1, 'Harsh':1.5})

# dropped_db = db.dropna()
# db = db.drop(['tower_height', 'single_blade_weight', 'project_ref','built_duration', 'nacelle_weights'],axis=1)


db['squared_power'] = (db['turbine_rating']/1000)**2


dropped_db = db[(db.operator == 'Competitor')] #& (db.type == 'gamma')]
# mock_db = dropped_db.drop(['blade_length'], axis=1)

# Alpha
# independent_var = ['water_depth', 'a1', 'turbine_rating', 'constant']
# mock_db = mock_db.drop(['squared_power','metocean'], axis=1)
# beta
# independent_var = ['a1', 'turbine_rating', 'constant']
# mock_db = mock_db.drop(['squared_power', 'water_depth', 'metocean'], axis=1)
# gammma
independent_var = ['blade_length', 'constant']
# mock_db = mock_db.drop(['squared_power', 'water_depth', 'metocean'], axis=1)
focussed_db = dropped_db[independent_var + [target]]

focussed_db = focussed_db.dropna()
without_target = focussed_db[independent_var]

# focussed_db = mock_db.drop(['operator', 'type', 'region'], axis=1)

x = without_target.to_numpy()
# independent_var = list(focussed_db.columns)
m = np.matmul(np.linalg.inv(np.matmul(x.transpose(), x)), x.transpose())
p = np.matmul(x,m)
observed = focussed_db[target].to_numpy()
weight_estimates = np.matmul(m,observed)

column_avgs = np.array([sum(focussed_db[column])/len(focussed_db) for column in independent_var])
normalised_weights = np.multiply(weight_estimates, column_avgs)

print(list(zip(independent_var, normalised_weights)))

estimates = np.matmul(p,observed)
residuals = np.matmul((np.identity(len(p))-p), observed)

variance_estimator = (np.linalg.norm(residuals))**2/len(observed)
print('Variance estimator for noise:', variance_estimator)

# inference
# find nan columns for blade length
empty_db = db[db[target].isna()]
# empty_db = empty_db[(empty_db['type'] == 'gamma') & (empty_db['operator'] == 'Competitor')]
empty_db = empty_db[(empty_db['operator'] == 'Competitor')]
relevant = empty_db[independent_var]
relevant = relevant.dropna()
print(relevant)
test_x = relevant.to_numpy()
predicted_values = np.matmul(test_x, weight_estimates)
relevant[target] = predicted_values

db.update(relevant)
db.to_excel("edited3.xlsx")  