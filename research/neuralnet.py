# -*- coding: utf-8 -*-

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', init='uniform'):
    print('Using NN with '+str(p.units)+' units per layer')
    model = Sequential()
    model.add(Dense(p.units, kernel_initializer = init, activation = 'relu', input_dim = X.shape[1]))
    model.add(Dense(p.units, kernel_initializer = init, activation = 'relu'))
    model.add(Dense(1, kernel_initializer = init, activation = 'sigmoid'))

	# Compile model
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def runNN1(conf):
    #from pandas import read_csv, set_option
    from sklearn.preprocessing import StandardScaler
#    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import accuracy_score
#    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from keras.wrappers.scikit_learn import KerasClassifier

    global ds
    global grid_result

    q.init(conf)
    ds = q.load_data()    

    #  Most used indicators: https://www.quantinsti.com/blog/indicators-build-trend-following-strategy/
    ds['date_to'] = ds['date'].shift(-1)
    # Calculate Features
    ds['VOL'] = ds['volumeto']/ds['volumeto'].rolling(window = p.vol_period).mean()
    ds['HH'] = ds['high']/ds['high'].rolling(window = p.hh_period).max() 
    ds['LL'] = ds['low']/ds['low'].rolling(window = p.ll_period).min()
    ds['DR'] = ds['close']/ds['close'].shift(1)
    ds['MA'] = ds['close']/ds['close'].rolling(window = p.sma_period).mean()
    ds['MA2'] = ds['close']/ds['close'].rolling(window = 2*p.sma_period).mean()
    ds['Std_dev']= ds['close'].rolling(p.std_period).std()/ds['close']
    ds['RSI'] = talib.RSI(ds['close'].values, timeperiod = p.rsi_period)
    ds['Williams %R'] = talib.WILLR(ds['high'].values, ds['low'].values, ds['close'].values, p.wil_period)
    
    # Tomorrow Return - this should not be included in training set
    ds['TR'] = ds['DR'].shift(-1)
    # Predicted value is whether price will rise
    ds['Price_Rise'] = np.where(ds['TR'] > 1, 1, 0)
    
    if p.max_bars > 0: ds = ds.tail(p.max_bars).reset_index(drop=True)
    ds = ds.dropna()

    # Separate input from output
    X = ds.iloc[:, -11:-2]
    y = ds.iloc[:, -1]
    
    # Separate train from test
    train_split = int(len(ds)*p.train_pct)
    test_split = int(len(ds)*p.test_pct)
    X_train, X_test, y_train, y_test = X[:train_split], X[-test_split:], y[:train_split], y[-test_split:]

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Cross Validation
    model = KerasClassifier(build_fn=create_model, verbose=0, epochs=100, batch_size=5)
#    estimators = []
#    estimators.append(('standardize', StandardScaler()))
#    estimators.append(('mlp', model))
#    pipeline = Pipeline(estimators)
#    results = cross_val_score(pipeline, X_train, y_train, cv=TimeSeriesSplit(n_splits=5))
#    print(results.mean())

    
    # Grid search epochs, batch size and optimizer
    # See: https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
#    optimizers = ['rmsprop', 'adam']
#    initial = ['glorot_uniform', 'normal', 'uniform']
#    epochs = [100]
#    batches = [10]
#    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=initial)
#    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=TimeSeriesSplit())
#    grid_result = grid.fit(X_train, y_train)

    # summarize results
#    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#    means = grid_result.cv_results_['mean_test_score']
#    stds = grid_result.cv_results_['std_test_score']
#    params = grid_result.cv_results_['params']
#    for mean, stdev, param in zip(means, stds, params):
#        print("%f (%f) with: %r" % (mean, stdev, param))

    models = []
    models.append(('NN', model))
    models.append(('LR' , LogisticRegression()))
    models.append(('LDA' , LinearDiscriminantAnalysis()))
    models.append(('KNN' , KNeighborsClassifier()))
    models.append(('CART' , DecisionTreeClassifier()))
    models.append(('NB' , GaussianNB()))
    models.append(('SVM' , SVC()))
    models.append(('RF' , RandomForestClassifier(n_estimators=50)))
    models.append(('XGBoost', XGBClassifier()))

    for name, model in models:
        clf = model
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred = y_pred.round()
        accu_score = accuracy_score(y_test, y_pred)
        print(name + ": " + str(accu_score))
