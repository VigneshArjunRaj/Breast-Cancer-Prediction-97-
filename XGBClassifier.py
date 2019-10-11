class Classifiers:
    def __init__(self,dataset,x,classval,a,b,normalize=True,encode=False,encodeval=0):
        global X,y,X_train,y_train,X_test,y_test
        if encode==True:
            from sklearn.preprocessing import LabelEncoder   
            le = LabelEncoder()
            encodeval1=data.columns[1 ]
            print(encodeval1)
            dataset[encodeval1]= le.fit_transform(dataset[encodeval1]) 
            from sklearn.preprocessing import OneHotEncoder 
            onehotencoder = OneHotEncoder(categorical_features = [0])
            global encoded
            encoded=dataset[encodeval1]
            encoded= onehotencoder.fit_transform(encoded).toarray()
            print(encoded)
            encoded=np.reshape(encoded,(np.size(encoded,axis=1),np.size(encoded,axis=0)))
            #print(temp)
            #print(dataset)
            X = dataset.iloc[:, a:b+1].values
            #X=np.delete(X,encodeval,axis=1)
            X=np.append(X,encoded,axis=1)
            y = dataset.iloc[:, classval].values
            print(X)
        else:
            X = dataset.iloc[:, a:b+1].values
            y = dataset.iloc[:, classval].values
        from sklearn.preprocessing import Imputer
        imputer = Imputer(missing_values = "NaN", strategy = 'mean', axis = 0)
        imputer = imputer.fit(X[:, a:b])
        X[:, a:b] = imputer.transform(X[:, a:b])
        X
        pca=PCA(n_components=2)
        X=pca.fit_transform(X)
        print(pca.explained_variance_ratio_)
        #from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
   def XGBR(self, max_depth=3, learning_rate=0.1, n_estimators=100, verbosity=1, silent=None, objective="binary:logistic", booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None):
        xgb=XGBClassifier(max_depth, learning_rate, n_estimators, verbosity, silent, objective, booster, n_jobs, nthread, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree, colsample_bylevel, colsample_bynode, reg_alpha, reg_lambda, scale_pos_weight, base_score, random_state, seed, missing)
        xgb.fit(X_train,y_train)
        y_pred=xgb.predict(X_test)
        acc=accuracy_score(y_test,y_pred)
        cm=confusion_matrix(y_test,y_pred)
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, xgb.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'brown',)))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('yellow','black'))(i), label = j)
        plt.title('XGBR (Test set)')
        plt.xlabel('pca1')
        plt.ylabel('pca2')
        plt.legend()
        plt.show()
        y_pred_prob=xgb.predict_proba(X_test)
        return y_pred,acc,cm,y_pred_prob
c=Classifiers(data="breast-cancer-wisconsin.data",x=40,classval=10,a=1,b=9)
c.XGBR()
