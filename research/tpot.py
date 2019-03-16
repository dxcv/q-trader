#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:27:47 2019

@author: igor
"""

#conda install numpy scipy scikit-learn pandas
#pip install deap update_checker tqdm stopit
#pip install xgboost
#pip install tpot
def tpot_test(conf):
    from tpot import TPOTRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import TimeSeriesSplit
    
    p.load_config(conf)
    ds = dl.load_price_data()
    ds = add_features(ds)

    X = ds[p.feature_list][:-1]
    y = ds['DR'].shift(-1)[:-1]

    # Split Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
    
    tpot = TPOTRegressor(n_jobs=-1, verbosity=2, max_time_mins=60, cv=TimeSeriesSplit(n_splits=3))
    
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('./tpot_out.py')
    
tpot_test('ETHUSDNN')
