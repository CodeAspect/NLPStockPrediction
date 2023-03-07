def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import json
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def make_classifier(X, y):
    # pl = Pipeline()
    clf = ElasticNetCV(n_estimators=10,
            max_depth=None,min_samples_split=2, random_state=0)
    return clf.fit(X, y)

def read_data(fname, sym, feature_subset=False, standardize=False, test_size=.2):
    ret = pd.read_csv(fname)
    # 2. Rename the keys that have the asset name im them
    ret.set_index("Date", inplace=True)

    if sym != "btc":
        _renamer = lambda sym, k: k[len(sym) + 1:].upper()\
                if sym in k else k 
        renamer = lambda sym: {k : _renamer(sym,k) for k in
            data_keys(sym)}
    # if "btc" != sym:
    #     ret = ret[data_keys(sym)]
        ret.rename(columns=renamer(sym), inplace=True)
    if standardize:
        tmp = MinMaxScaler().fit_transform(
                ret.drop(columns=['groundTruth']))
        tmp = pd.DataFrame(tmp, index=ret.drop(
            columns=['groundTruth']).index)
        cols = dict(zip(tmp.keys(), ret.drop(columns=['groundTruth'], axis=1).keys().values))
        tmp.rename(columns=cols, inplace=True)
        ret = tmp.join(ret[['groundTruth']])
    train, test = train_test_split(ret, shuffle=False, test_size=test_size)
    Xtrain, ytrain = train.drop("groundTruth", axis=1), train[['groundTruth']]
    Xtest, ytest = test.drop("groundTruth", axis=1), test[['groundTruth']]


    return Xtrain, ytrain, Xtest, ytest

def read_btc(feature_subset=False, standardize=False):
    Xtrain_btc, ytrain_btc, Xtest_btc, ytest_btc = read_data('./data/BTCTweetData_Final.csv', "btc",
            feature_subset=feature_subset, standardize=standardize)

    Xtrain_btc2, ytrain_btc2, Xtest_btc2, ytest_btc2 = read_data('./data/GuardianBTCNews_Final.csv', "btc",
            feature_subset=feature_subset, standardize=standardize)
    Xtrain_btc = pd.concat([Xtrain_btc, Xtrain_btc2])
    ytrain_btc = pd.concat([ytrain_btc, ytrain_btc2])
    Xtest_btc = pd.concat([Xtest_btc, Xtest_btc2])
    ytest_btc = pd.concat([ytest_btc, ytest_btc2])

    return Xtrain_btc, ytrain_btc, Xtest_btc, ytest_btc

def read_amzn(feature_subset=False, standardize=False):
    return read_data('./data/Amazon_Final.csv', "amzn",
            feature_subset=feature_subset, standardize=standardize)

def read_aapl(feature_subset=False, standardize=False):
    return read_data('./data/Apple_Final.csv', "aapl",
            feature_subset=feature_subset, standardize=standardize)

def read_gme(feature_subset=False, standardize=False):
    return read_data('./data/GME_Final.csv', "gme",
            feature_subset=feature_subset, standardize=standardize)



def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(34, 21))

    axes[0].set_title(title)

    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    print("Test Scores Mean")
    print(test_scores_mean)
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def make_graphs(df, clf, sym="amzn", ext="pdf"):
    plt.rcParams["figure.figsize"] = (20,5)
    print("len(df.keys().values)={}".format(len(df.keys().values)))
    keys = [a for a in df.keys().values[:4]] + ['y_pred', 'groundTruth']
    df[keys].rolling(10, axis=0).mean().plot()
    plt.title("{}: Semantic Features".format(sym))
    plt.savefig("./results/{}.{}SemanticFeatures.{}".format(clf, sym, ext))
    plt.close()
    
    keys = [a for a in df.keys().values[4:8]] + ['y_pred', 'groundTruth']
    df[keys].rolling(5, axis=0).mean().plot()
    plt.title("{}: Tech Features 1".format(sym))
    plt.savefig("./results/{}.{}TechFeatures1.{}".format(clf, sym, ext))
    plt.close()
    
    keys = [a for a in df.keys().values[8:12]]+ ['y_pred', 'groundTruth']
    df[keys].rolling(5, axis=0).mean().plot()
    plt.title("{}: Tech Features 2".format(sym))
    plt.savefig("./results/{}.{}TechFeatures2.{}".format(clf, sym, ext))
    plt.close()

    keys = [a for a in df.keys().values[12:16]]+ ['y_pred', 'groundTruth']
    df[keys].rolling(5, axis=0).mean().plot()
    plt.title("{}: Tech Features 3".format(sym))
    plt.savefig("./results/{}.{}TechFeatures3.{}".format(clf, sym, ext))
    plt.close()

    keys = [a for a in df.keys().values[16:20]]+ ['y_pred', 'groundTruth']
    df[keys].rolling(5, axis=0).mean().plot()
    plt.title("{}: Tech Features 4".format(sym))
    plt.savefig("./results/{}.{}TechFeatures4.{}".format(clf, sym, ext))
    plt.close()

    keys = [a for a in df.keys().values[20:24]]+ ['y_pred', 'groundTruth']
    df[keys].rolling(5, axis=0).mean().plot()
    plt.title("{}: Tech Features 5".format(sym))
    plt.savefig("./results/{}.{}.TechFeatures5.{}".format(clf, sym, ext))
    plt.close()

    keys = [a for a in df.keys().values[24:28]]+ ['y_pred', 'groundTruth']
    df[keys].rolling(5, axis=0).mean().plot()
    plt.title("{}: Tech Features 6".format(sym))
    plt.savefig("./results/{}.{}.TechFeatures6.{}".format(clf, sym, ext))
    plt.close()

    keys = [a for a in df.keys().values[28:]]#+ ['y_pred', 'groundTruth']
    df[keys].rolling(5, axis=0).mean().plot()
    plt.title("{}: Tech Features 7".format(sym))
    plt.savefig("./results/{}.{}.TechFeatures7.{}".format(clf, sym, ext))
    plt.close()

    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

    return 

def do_elastic_net(features):
    results, Xamzn, Xaapl, Xgme, Xbtc = run_combined_data(ElasticNetCV, features, discretize=True)
    make_graphs(Xamzn, sym="amzn")
    make_graphs(Xaapl, sym="aapl")
    make_graphs(Xgme, sym="gme")
    make_graphs(Xbtc, sym="btc")
    return results

def select_features(X, y):
    # sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
    # sel.fit(X, y)
    # return X.columns[(sel.get_support())]
    """ 
    print("clf.indeces={}".format(clf.get_support()))
    print(X)
    print(X_new)
    tmp = pd.DataFrame(X_new, X.index)
    print(tmp)
    tmp.rename(columns={i : X.keys()[i]
        for i in range(len(X.keys()))}, inplace=True)
    print(tmp)
    """
    print("select_features(X, y)")
    sel = SelectKBest(chi2, k=20)
    sel.fit(X, y)
    cols = X.columns[(sel.get_support())].append(X.columns[-9:]).append(X.columns[:3])
    print(cols)
    return cols

def do_randomforest(X, y, features, do_LC=False, ext='pdf'):
    if do_LC:
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
        title = "Learning Curve (Random Forest)"
        estimator = RandomForestClassifier()
        lc = plot_learning_curve(estimator, title,
                X, y, axes=axes, ylim=(0.0, 1.01), cv=cv, n_jobs=4)
        lc.savefig("results/rfLearningCurves.{}".format(ext))

    results, Xamzn, Xaapl, Xgme, Xbtc = run_combined_data(RandomForestClassifier, features, do_LR=True, discretize=True)
    if True:
        make_graphs(Xamzn, "rf", sym="amzn")
        make_graphs(Xaapl, "rf", sym="aapl")
        make_graphs(Xgme, "rf", sym="gme")
        make_graphs(Xbtc, "rf", sym="btc")
    return results

def do_randomforest_roc(X_train, y_train, X_test, y_test, ext='pdf'):
    mlp = RandomForestClassifier().fit(X_train, y_train)
    metrics.plot_roc_curve(mlp, X_test, y_test)
    plt.title("RoC (Random Forest)")
    plt.savefig("./results/rocRandomForest.{}".format(ext))

def do_extratrees(X, y, features, do_LC=False, ext='pdf'):
    if do_LC:
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
        title = "Learning Curve (ExtraTrees)"
        estimator = ExtraTreesClassifier()
        lc = plot_learning_curve(estimator, title,
                X, y, axes=axes, ylim=(0.0, 1.01), cv=cv, n_jobs=4)
        lc.savefig("results/etLearningCurves.{}".format(ext))

    results, Xamzn, Xaapl, Xgme, Xbtc = run_combined_data(ExtraTreesClassifier, features, do_LR=True)
    make_graphs(Xamzn, "et", sym="amzn")
    make_graphs(Xaapl, "et", sym="aapl")
    make_graphs(Xgme, "et", sym="gme")
    make_graphs(Xbtc, "et", sym="btc")
    return results

def do_extratrees_roc(X_train, y_train, X_test, y_test, ext='pdf'):
    print(X_train)
    mlp = ExtraTreesClassifier().fit(X_train, y_train)
    metrics.plot_roc_curve(mlp, X_test, y_test)
    plt.title("RoC (Extra Trees)")
    plt.savefig("./results/rocExtraTrees.{}".format(ext))

def do_mlp_roc(X_train, y_train, X_test, y_test):
    mlp = MLPClassifier().fit(X_train, y_train)
    metrics.plot_roc_curve(mlp, X_test, y_test)
    plt.title("RoC (Neural Network)")
    plt.savefig("./results/rocMLP.pdf")

def do_mlp(X, y, features, do_LC=True):
    if do_LC:
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
        title = "Learning Curve (Neural Network)"
        estimator = MLPClassifier()
        lc = plot_learning_curve(estimator, title,
                X, y, axes=axes, ylim=(0.0, 1.01), cv=cv, n_jobs=4)
        lc.savefig("results/mlpLearningCurves.pdf")
    results, Xamzn, Xaapl, Xgme, Xbtc = run_combined_data(
            MLPClassifier, features, do_LR=True)
    if True:
        make_graphs(Xamzn, "mlp", sym="amzn")
        make_graphs(Xaapl, "mlp", sym="aapl")
        make_graphs(Xgme, "mlp", sym="gme")
        make_graphs(Xbtc, "mlp", sym="btc")
    return results


def do_svc_roc(X_train, y_train, X_test, y_test, ext="pdf"):
    svc = SVC().fit(X_train, y_train)
    metrics.plot_roc_curve(svc, X_test, y_test)
    plt.title("RoC (SVC)")
    plt.savefig("./results/rocSVC.{}".format(ext))

def do_svc(X, y, features, do_LC=False, ext="pdf"):
    if do_LC:
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
        title = "Learning Curve (Linear-SVC)"
        estimator = SVC(kernel='linear')
        lc = plot_learning_curve(estimator, title,
                X, y, axes=axes, ylim=(0.0, 1.01), cv=cv, n_jobs=4)
        lc.savefig("results/svcLearningCurves.{}".format(ext))
    results, Xamzn, Xaapl, Xgme, Xbtc = run_combined_data(SVC, features, do_LR=True)
    if True:
        make_graphs(Xamzn, "svc", sym="amzn")
        make_graphs(Xaapl, "svc", sym="aapl")
        make_graphs(Xgme, "svc", sym="gme")
        make_graphs(Xbtc, "svc", sym="btc")
    return results


def make_combined_dataset(feature_subset=False, standardize=True):

    # Load the data
    Xtrain_amzn, ytrain_amzn, Xtest_amzn, ytest_amzn = read_amzn(
            feature_subset=feature_subset, standardize=standardize)
    Xtrain_aapl, ytrain_aapl, Xtest_aapl, ytest_aapl = read_aapl(
            feature_subset=feature_subset, standardize=standardize)
    Xtrain_gme, ytrain_gme, Xtest_gme, ytest_gme = read_gme(
            feature_subset=feature_subset, standardize=standardize)
    Xtrain_btc, ytrain_btc, Xtest_btc, ytest_btc = read_btc(
            feature_subset=feature_subset, standardize=standardize)
    Xcombined = pd.concat([Xtrain_amzn, Xtrain_aapl,
        Xtrain_gme, Xtest_amzn, Xtest_aapl,
        Xtest_gme, Xtrain_btc])
    ycombined = pd.concat([ytrain_amzn, ytrain_aapl,
        ytrain_gme, ytest_amzn, ytest_aapl, ytest_gme, ytrain_btc])


    if True:
        print(Xcombined)
        features = select_features(Xcombined, ycombined)
        print(features)
        print(Xcombined[features].keys())
        Xcombined = Xcombined[features]
    
    return Xcombined, ycombined, features

def make_train_test_combined(features, feature_subset=False,
        standardize=True):

    # Load the data
    Xtrain_amzn, ytrain_amzn, Xtest_amzn, ytest_amzn = read_amzn(
            feature_subset=feature_subset, standardize=standardize)
    Xtrain_aapl, ytrain_aapl, Xtest_aapl, ytest_aapl = read_aapl(
            feature_subset=feature_subset, standardize=standardize)
    Xtrain_gme, ytrain_gme, Xtest_gme, ytest_gme = read_gme(
            feature_subset=feature_subset, standardize=standardize)
    Xtrain_btc, ytrain_btc, Xtest_btc, ytest_btc = read_btc(
            feature_subset=feature_subset, standardize=standardize)
    Xcombined = pd.concat([Xtrain_amzn, Xtrain_aapl, Xtrain_gme, Xtest_amzn, Xtest_aapl, Xtest_gme, Xtrain_btc])
    ycombined = pd.concat([ytrain_amzn, ytrain_aapl, ytrain_gme, ytest_amzn, ytest_aapl, ytest_gme, ytrain_btc])
    Xcombined_test = pd.concat([Xtest_amzn, Xtest_aapl, Xtest_gme, Xtrain_btc])
    ycombined_test = pd.concat([ytest_amzn, ytest_aapl, ytest_gme, ytrain_btc])
    if True:
        Xcombined_test = Xcombined_test[features]
        Xcombined = Xcombined[features]
    return Xcombined, ycombined, Xcombined_test, ycombined_test



def run_combined_data(clf, features, do_LR=False, discretize=False, feature_subset=False, standardize=True):

    # Load the data
    Xtrain_amzn, ytrain_amzn, Xtest_amzn, ytest_amzn = read_amzn(
            feature_subset=feature_subset, standardize=standardize)
    Xtrain_aapl, ytrain_aapl, Xtest_aapl, ytest_aapl = read_aapl(
            feature_subset=feature_subset, standardize=standardize)
    Xtrain_gme, ytrain_gme, Xtest_gme, ytest_gme = read_gme(
            feature_subset=feature_subset, standardize=standardize)
    Xtrain_btc, ytrain_btc, Xtest_btc, ytest_btc = read_btc(
            feature_subset=feature_subset, standardize=standardize)

    Xtrain_amzn = Xtrain_amzn[features]
    Xtest_amzn = Xtest_amzn[features]
    Xtrain_aapl = Xtrain_aapl[features]
    Xtest_aapl = Xtest_aapl[features]
    Xtrain_gme = Xtrain_gme[features]
    Xtest_gme = Xtest_gme[features]
    Xtrain_btc = Xtrain_btc[features]
    Xtest_btc = Xtest_btc[features]

    # rawBTCTweet
    # RawBTCStock
    Xcombined = pd.concat([Xtrain_amzn, Xtrain_aapl, Xtrain_gme, Xtrain_btc])
    ycombined = pd.concat([ytrain_amzn, ytrain_aapl, ytrain_gme, ytrain_btc])


    cclf = clf().fit(Xcombined, ycombined)
    ## Do stuff now:
    print("Totel: Xcombined={}, ycombined={}".format(Xcombined.shape, ycombined.shape))
    ypred_amzn = cclf.predict(Xtest_amzn)
    ypred_amzn = pd.DataFrame(ypred_amzn, index=ytest_amzn.index)
    if discretize:
        ypred_amzn = ypred_amzn.applymap(lambda x: 1 if x > 0.446 else 0)
    DEBUG = False
    if DEBUG:
        print(ytest_amzn.values)
        print(ypred_amzn)
    results = {}
    results['Amazon'] = {
            "results" : classification_report(ytest_amzn,
                ypred_amzn, output_dict=True), 
            "confusion_matrix" : confusion_matrix(
                ytest_amzn, ypred_amzn).tolist()
            } 
    ypred_aapl = cclf.predict(Xtest_aapl)
    if DEBUG:
        print(ytest_aapl.values)
        print(ypred_aapl)
    # Ideally this threshold is selected to minimize the MSE (or minimize the cosine distance between y_test and y_pred)

    ypred_aapl = pd.DataFrame(ypred_aapl, index=ytest_aapl.index)
    if discretize:
        ypred_aapl = ypred_aapl.applymap(lambda x: 1 if x > 0.31 else 0)


    results['Apple'] = {
            "results" : classification_report(ytest_aapl,
                ypred_aapl, output_dict=True), 
            "confusion_matrix" : confusion_matrix(ytest_aapl,
                ypred_aapl).tolist()
            } 

    print("GME: Classification results")
    ypred_gme = cclf.predict(Xtest_gme)
    if DEBUG:
        print(ypred_gme)
        print(ytest)
    ypred_gme = pd.DataFrame(ypred_gme,index=ytest_gme.index)
    if discretize:
        ypred_gme = ypred_gme.applymap(lambda x: 0 if x > 0.446 else 1)
    if DEBUG:
        print(ypred_gme)
    results['GME'] = {
            "results" : classification_report(ytest_gme,
                ypred_gme, output_dict=True), 
            "confusion_matrix" : confusion_matrix(
                ytest_gme, ypred_gme).tolist()
            } 
    ypred_btc = cclf.predict(Xtest_btc)
    ypred_btc = pd.DataFrame(ypred_btc,index=ytest_btc.index)
    results['BTC'] = {
            "results" : classification_report(ytest_btc,
                ypred_btc, output_dict=True), 
            "confusion_matrix" : confusion_matrix(
                ytest_btc, ypred_btc).tolist()
            } 

    if do_LR:
        ## Calling Model ##
        logReg = clf()
        """ 
                penalty="l1",
                solver="liblinear",
                warm_start=True,
                intercept_scaling=10000.0,)
        """ 

        ## 10 Fold Cross Validiation ##
        kf = KFold(n_splits=10, random_state=True, shuffle=True)
        kf.get_n_splits(Xcombined)

        score = []
        accuracy = []
        f1Score = []
        recall = []


        for train_index, test_index in kf.split(Xcombined):
            x_train, x_test = Xcombined.iloc[train_index], Xcombined.iloc[test_index]
            y_train, y_test = ycombined.iloc[train_index], ycombined.iloc[test_index]

            logReg.fit(x_train, y_train)

            y_pred = logReg.predict(x_test)
            report = classification_report(y_test, y_pred, output_dict=True)

            score.append(logReg.score(x_test, y_test))
            accuracy.append(report['accuracy'])
            f1Score.append(report['macro avg']['f1-score'])
            recall.append(report['macro avg']['recall'])

        ## Displaying Results ##
        print("{}".format(clf))
        print("Score: ", np.mean(score))
        print("Accuracy: ", np.mean(accuracy))
        print("F1 Score: ", np.mean(f1Score))
        print("Recall: ", np.mean(recall))
    Xtest_amzn = Xtest_amzn.join(ypred_amzn)
    Xtest_amzn.rename(columns={0 : 'y_pred'}, inplace=True)
    Xtest_amzn = Xtest_amzn.join(ytest_amzn)
    Xtest_aapl = Xtest_aapl.join(ypred_aapl)
    Xtest_aapl.rename(columns={0 : 'y_pred'}, inplace=True)
    Xtest_aapl = Xtest_aapl.join(ytest_aapl)
    Xtest_gme = Xtest_gme.join(ypred_gme)
    Xtest_gme.rename(columns={0 : 'y_pred'}, inplace=True)
    Xtest_gme = Xtest_gme.join(ytest_gme)
    Xtest_btc = Xtest_btc.join(ypred_btc)
    Xtest_btc.rename(columns={0 : 'y_pred'}, inplace=True)
    Xtest_btc = Xtest_btc.join(ytest_btc)

    return (results, Xtest_amzn,
            Xtest_aapl,
            Xtest_gme,
            Xtest_btc)

def test_technical_features():

    Xg = Xg[data_keys('gme')]
    yg = yg[data_keys('gme')]

    Xam = Xam[data_keys('amzn')]
    yam = yam[data_keys('amzn')]
    
    X = X[data_keys('aapl')]
    y = y[data_keys('aapl')]

def do_gme():
    X, y = read_amzn()
    X = X[data_keys('gme')]

def sentiment_features():
    """ """
    return ['SA_NLTK_compound',
            'SA_TextBlob',
            'SA_FD'] 

def technical_features(sym):
    """
    atr: Average True Range
    bb_bbl: Bollinger Band Low
    bb_bbh: Bollinger Band High
    bb_bbm: Bollinger Band Mean


    rsi: RSI Indicator: 

    Traditional interpretation and usage of the RSI are that
    values of 70 or above indicate that a security is becoming
    overbought or overvalued and may be primed for a trend
    reversal or corrective pullback in price. An RSI reading
    of 30 or below indicates an oversold or undervalued
    condition.

    macd_signal: Moving Average Convergence Divergence Signal
    macd_diff: Moving Average Convergence Divergence Difference
    macd: Moving Average Convergence Divergence

    Traders may buy the security when the MACD crosses above
    its signal line and sell—or short—the security when the
    MACD crosses below the signal line. Moving average
    convergence divergence (MACD) indicators can be interpreted
    in several ways, but the more common methods are crossovers,
    divergences, and rapid rises/falls.

    volume: Trade volume
    """
    return ['{}:atr'.format(sym),
            '{}:bb_bbl'.format(sym),
            '{}:bb_bbh'.format(sym),
            '{}:bb_bbm'.format(sym),
            '{}:rsi'.format(sym),
            '{}:macd_signal'.format(sym),
            '{}:macd_diff'.format(sym),
            '{}:macd'.format(sym),
            '{}:volume'.format(sym)]


def standardize_data(data):
    df = StandardScaler().fit_transform(data.drop(columns=['groundTruth']))
    df = pd.DataFrame(df).join(data['groundTruth'])

def get_label():
    return ['groundTruth']

def data_keys(sym='amzn'):
    keys = ['SA_NLTK_compound',
            'SA_TextBlob',
            'SA_FD']\
                    + technical_features(sym)\
                    +['groundTruth']
    return keys

def btc_data_keys():
    btc_tech = ['atr'.upper(),
            'bb_bbl'.upper(),
            'bb_bbh'.upper(),
            'bb_bbm'.upper(),
            'rsi'.upper(),
            'MACD_Signal',
            'MACD_Diff',
            'MACD',
            'Volume']
    keys = ['SA_NLTK_compound',
            'SA_TextBlob',
            'SA_FD'] + btc_tech + ['groundTruth']
    return keys

def main():
    """ """
    # do_elastic_net()
    # Xamzn, Xaapl, Xgme, Xbtc = run_combined_data(ExtraTreesClassifier(n_estimators=10))

    X, y, features = make_combined_dataset()
    X_train, y_train, X_test, y_test = make_train_test_combined(features)
    results = {}
    do_ET=False
    do_LC=True
    if do_ET:
        do_extratrees_roc(X_train, y_train, X_test, y_test)
        result = do_extratrees(X, y, features, do_LC=do_LC)
        results['extra-trees'] = result

    do_SCV=True
    if do_SCV:
        result = do_svc(X, y, features, do_LC=do_LC)
        do_svc_roc(X_train, y_train, X_test, y_test)
        results['svm'] = result

    do_MLP=True
    if do_MLP:
        do_mlp_roc(X_train, y_train, X_test, y_test)
        result = do_mlp(X, y, features, do_LC=do_LC)
        results['mlp'] = result
    do_RF=True
    if do_RF:
        do_randomforest_roc(X_train, y_train, X_test, y_test)
        result = do_randomforest(X, y, features, do_LC=do_LC)
        results['random-forest'] = result

    if True:
        print(results)
        with open("results/results.json", "w") as f:
            json.dump(results, f)

if __name__ == '__main__':
    main()
