def implement_ml(X, label_list, sfs=True,svc=True, knn=True, k_range= range(13,14)):
    '''
    Function to implement various ML algorithms on processed data.
    Splits data into test and train (2/3,1/3)
    :param X: Processed data
    :param label_list: Labels corresponding to data
    :return: TBD
    '''

    # Split into test and train:
    index = np.int32(np.floor(X.shape[0] * (2 / 3)))  # Index up to 2/3 of datapoints.

    X_train = X.iloc[0:index, :]  # First 2/3 data points
    label_train = label_list.iloc[0:index]

    X_test = X.iloc[index:X.shape[0], :]
    label_test = label_list.iloc[index:label_list.shape[0]]

    # Implement sfs:

    if sfs:

        start = timer.time()
        # Apply feature selection

        # Initialize lists:
        if svc:
            k_list_svc = list()
            f_1_svc = list()
            parameters_svc = list()
        if knn:
            k_list_knn = list()
            f_1_knn = list()
            parameters_knn = list()

        # Apply SFS
        for k in k_range:

            df_train_fs = X_train[:]

            print(f"Applying SFS with {k} dimensions")
            # Apply sequential feature selection keeping k dimensions:
            rdg_cls = RidgeClassifier(class_weight='balanced')
            sfs = SequentialFeatureSelector(rdg_cls, n_features_to_select=k, scoring='f1')
            sfs.fit(df_train_fs, np.array(label_train).ravel())

            # Apply to test and train:
            df_train_fs = sfs.transform(df_train_fs)
            df_train_fs = pd.DataFrame(df_train_fs, columns=sfs.get_feature_names_out())

            # SVC Classifier, perform grid-search for each k:
            if svc:

                params = [
                    {'standardscaler': ['passthrough', StandardScaler()],
                     'svc__C': [1, 10, 100, 1000], 'svc__kernel': ['rbf'], 'svc__gamma': [0.001, 0.0001],
                     'svc__class_weight': [None, 'balanced']},
                    {'standardscaler': ['passthrough', StandardScaler(), MinMaxScaler()],
                     'svc__C': [1, 10, 100, 1000], 'svc__kernel': ['linear'], 'svc__class_weight': [None, 'balanced']}
                ]

                pipe = make_pipeline(StandardScaler(), SVC())
                gs = GridSearchCV(pipe, params, scoring='f1')
                gs.fit(df_train_fs, np.array(label_train).ravel())

                # Store best f1 score for each dimension k:
                k_list_svc.append(k)
                f_1_svc.append(gs.best_score_)
                parameters_svc.append(gs.best_params_)

            # KNN classifier, perform grid-search for each k:
            if knn:

                params = [
                    {'kneighborsclassifier__n_neighbors': (7, 8, 9, 10),  # 8,9,10
                     'kneighborsclassifier__weights': ('uniform', 'distance'),
                     'kneighborsclassifier__metric': ('minkowski', 'chebyshev')}
                ]

                pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
                gs = GridSearchCV(pipe, params, scoring='f1')
                gs.fit(df_train_fs, np.array(label_train).ravel())

                # Store best f1 score for each dimension k:
                k_list_knn.append(k)
                f_1_knn.append(gs.best_score_)
                parameters_knn.append(gs.best_params_)

        else:
            print('Please select a classifier from SVC or KNN')

        end = timer.time()

        print(f'Elapsed time for SFS is {end - start} seconds')


    # find the dimension that gives the best f-1 score and corresponding parameters for all classifiers:

    if svc:

        best_dim_svc = k_list_svc[np.array(f_1_svc).argmax()]  # Retrieve best dimension
        best_params_svc = parameters_svc[np.array(f_1_svc).argmax()]  # Retrieve best parameters

        print(f'For the SVC classifier, '
              f'The optimal number of dimension is {best_dim_svc} with'
              f' an F1 score of {np.round(np.array(f_1_svc).max(), 3)}'
              f' with parameters {best_params_svc}')



        # Train with optimal parameters and dimensions:
        print('Training the SVC classifier with optimal parameters')
        C_ = best_params_svc['svc__C']
        kernel_ = best_params_svc['svc__kernel']
        weight_ = best_params_svc['svc__class_weight']


        if best_params_svc['svc__kernel'] == 'linear':
            pipe = make_pipeline(SequentialFeatureSelector(rdg_cls, n_features_to_select=best_dim_svc, scoring='f1'),
                                 SVC(C=C_, kernel=kernel_, class_weight=weight_))

        if best_params_svc['svc__kernel'] == 'rbf':
            pipe = make_pipeline(SequentialFeatureSelector(rdg_cls, n_features_to_select=best_dim_svc, scoring='f1'),
                                 SVC(C=C_, kernel=kernel_,
                                     gamma=best_params_svc['svc__gamma'],
                                     class_weight=weight_))

        # Fit model:
        pipe.fit(X_train, np.array(label_train).ravel())

        # Evaluate on test set:
        pred = pipe.predict(X_test)
        f1_score = metrics.f1_score(np.array(label_test), pred)
        acc = metrics.accuracy_score(np.array(label_test), pred)
        print(f'The F1 score on the test set with the SVC classifier is {f1_score} and the accuracy is {acc}')

        # Confusion matrix:
        cm = metrics.confusion_matrix(np.array(label_test), pred)
        cm_display = metrics.ConfusionMatrixDisplay(cm).plot()
        plt.title("Confusion Matrix for SVC Classifier")
        plt.show()

    if knn:

        best_dim_knn = k_list_knn[np.array(f_1_knn).argmax()]  # Retrieve best dimension
        best_params_knn = parameters_knn[np.array(f_1_knn).argmax()]  # Retrieve best parameters

        print(f'For the KNN classifier, '
              f'The optimal number of dimension is {best_dim_knn} with'
              f' an F1 score of {np.round(np.array(f_1_knn).max(), 3)}'
              f' with parameters {best_params_knn}')

        # Train with best parameters and plot confusion matrix:
        print('Training the KNN classifier with optimal parameters')
        metric_ = best_params_knn['kneighborsclassifier__metric']
        n_neighbors_ = best_params_knn['kneighborsclassifier__n_neighbors']
        weights_ = best_params_knn['kneighborsclassifier__weights']

        pipe = make_pipeline(SequentialFeatureSelector(rdg_cls, n_features_to_select=best_dim_knn, scoring='f1'),
                             KNeighborsClassifier(metric=metric_, n_neighbors=n_neighbors_,
                                                  weights=weights_))
        # Fit model:
        pipe.fit(X_train, np.array(label_train).ravel())

        # Evaluate on test set:
        pred = pipe.predict(X_test)
        f1_score = metrics.f1_score(np.array(label_test), pred)
        acc = metrics.accuracy_score(np.array(label_test), pred)
        print(f'The F1 score on the test set with the KNN classifier is {f1_score} and the accuracy is {acc}')

        # Confusion matrix:
        cm = metrics.confusion_matrix(np.array(label_test), pred)
        cm_display = metrics.ConfusionMatrixDisplay(cm).plot()
        plt.title("Confusion Matrix for KNN Classifier")
        plt.show()

