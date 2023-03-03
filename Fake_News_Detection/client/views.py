


import itertools

from django.db.models import Q
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc




from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style



from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import pandas as pd
import string


from django.shortcuts import render, redirect

# Create your views here.

from client.models import FakeRealModel, AccuracyModel, AccuracyModel1, AccuracyModel2,clientinformation
from management.models import AdminModel


def login(request):
    if request.method=="POST":
        usid=request.POST.get('username')
        pswd = request.POST.get('password')
        try:
            check = clientinformation.objects.get(userid=usid,password=pswd)
            request.session['uid']=check.id
            return redirect('mydetails')
        except:
            pass
    return render(request,'client/login.html')

def register(request):
    if request.method == "POST":
        firstname = request.POST.get('firstname')
        lastname = request.POST.get('lastname')
        uname = request.POST.get('uname')
        password = request.POST.get('password')
        phone = request.POST.get('phone')
        email = request.POST.get('email')
        gender = request.POST.get('gender')


        clientinformation.objects.create(firstname=firstname, lastname=lastname, userid=uname, password=password,
                                     phoneno=phone,email=email,gender=gender)
        return redirect('login')
    return render(request, 'client/register.html')

def analysis(request):


    df = pd.read_csv('Detecting-Fake-News-with-Scikit-Learn-master/data/fake_or_real_news.csv')

    # Inspect shape of `df`
    df.shape

    # Print first lines of `df`
    df.head()

    # Set index

    # Print first lines of `df`
    df.head()

    # Set `y`
    y = df.label

    # Drop the `label` column
    df.drop("label", axis=1)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

    # Initialize the `count_vectorizer`
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(X_train)

    # Transform the test set
    count_test = count_vectorizer.transform(X_test)

    # Initialize the `tfidf_vectorizer`
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Fit and transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test set
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Get the feature names of `tfidf_vectorizer`
    print(tfidf_vectorizer.get_feature_names()[-10:])

    # Get the feature names of `count_vectorizer`
    print(count_vectorizer.get_feature_names()[:10])

    count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

    difference = set(count_df.columns) - set(tfidf_df.columns)


    set()

    print(count_df.equals(tfidf_df))

    count_df.head()

    tfidf_df.head()

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        See full source and example:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    clf = MultinomialNB()

    clf.fit(tfidf_train, y_train)
    pred = clf.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    accuracy =("accuracy:   %0.3f" % score)

    cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

    AccuracyModel.objects.create(accuracy=accuracy)

    objs = AccuracyModel.objects.filter(Q(id='1'))
    return render(request, 'client/analysis.html',{'v':objs})

def mydetails(request):
    userid = request.session['uid']

    ted = clientinformation.objects.get(id=userid)

    return render(request, 'client/mydetails.html',{'objects':ted})

def analysis1(request):


    df = pd.read_csv('Detecting-Fake-News-with-Scikit-Learn-master/data/fake_or_real_news.csv')

    # Inspect shape of `df`
    df.shape

    # Print first lines of `df`
    df.head()

    # Set index

    # Print first lines of `df`
    df.head()

    # Set `y`
    y = df.label

    # Drop the `label` column
    df.drop("label", axis=1)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

    # Initialize the `count_vectorizer`
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(X_train)

    # Transform the test set
    count_test = count_vectorizer.transform(X_test)

    # Initialize the `tfidf_vectorizer`
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Fit and transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test set
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Get the feature names of `tfidf_vectorizer`
    print(tfidf_vectorizer.get_feature_names()[-10:])

    # Get the feature names of `count_vectorizer`
    print(count_vectorizer.get_feature_names()[:10])

    count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

    difference = set(count_df.columns) - set(tfidf_df.columns)


    set()

    print(count_df.equals(tfidf_df))

    count_df.head()

    tfidf_df.head()

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        See full source and example:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    clf = MultinomialNB()

    clf.fit(count_train, y_train)
    pred = clf.predict(count_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    accuracy1=("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

    AccuracyModel1.objects.create(accuracy1=accuracy1)
    objs1 = AccuracyModel1.objects.filter(Q(id='1'))

    return render(request, 'client/analysis1.html',{'v':objs1})

def analysis2(request):


    df = pd.read_csv('Detecting-Fake-News-with-Scikit-Learn-master/data/fake_or_real_news.csv')

    # Inspect shape of `df`
    df.shape

    # Print first lines of `df`
    df.head()

    # Set index

    # Print first lines of `df`
    df.head()

    # Set `y`
    y = df.label

    # Drop the `label` column
    df.drop("label", axis=1)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

    # Initialize the `count_vectorizer`
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(X_train)

    # Transform the test set
    count_test = count_vectorizer.transform(X_test)

    # Initialize the `tfidf_vectorizer`
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Fit and transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test set
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Get the feature names of `tfidf_vectorizer`
    print(tfidf_vectorizer.get_feature_names()[-10:])

    # Get the feature names of `count_vectorizer`
    print(count_vectorizer.get_feature_names()[:10])

    count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

    difference = set(count_df.columns) - set(tfidf_df.columns)


    set()

    print(count_df.equals(tfidf_df))

    count_df.head()

    tfidf_df.head()

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        See full source and example:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    linear_clf = PassiveAggressiveClassifier()

    linear_clf.fit(tfidf_train, y_train)
    pred = linear_clf.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    accuracy2=("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

    AccuracyModel2.objects.create(accuracy2=accuracy2)
    objs2 = AccuracyModel2.objects.filter(Q(id='1'))

    return render(request, 'client/analysis2.html',{'v':objs2})

def analysis3(request):


    df = pd.read_csv('Detecting-Fake-News-with-Scikit-Learn-master/data/fake_or_real_news.csv')

    # Inspect shape of `df`
    df.shape

    # Print first lines of `df`
    df.head()

    # Set index

    # Print first lines of `df`
    df.head()

    # Set `y`
    y = df.label

    # Drop the `label` column
    df.drop("label", axis=1)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

    # Initialize the `count_vectorizer`
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(X_train)

    # Transform the test set
    count_test = count_vectorizer.transform(X_test)

    # Initialize the `tfidf_vectorizer`
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Fit and transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test set
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Get the feature names of `tfidf_vectorizer`
    print(tfidf_vectorizer.get_feature_names()[-10:])

    # Get the feature names of `count_vectorizer`
    print(count_vectorizer.get_feature_names()[:10])

    count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

    difference = set(count_df.columns) - set(tfidf_df.columns)


    set()

    print(count_df.equals(tfidf_df))

    count_df.head()

    tfidf_df.head()

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        See full source and example:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        clf = MultinomialNB()

        clf.fit(tfidf_train, y_train)
        pred = clf.predict(tfidf_test)
        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)
        cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
        plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

    clf = MultinomialNB(alpha=0.1)

    last_score = 0
    for alpha in np.arange(0, 1, .1):
        nb_classifier = MultinomialNB(alpha=alpha)
        nb_classifier.fit(tfidf_train, y_train)
        pred = nb_classifier.predict(tfidf_test)
        score = metrics.accuracy_score(y_test, pred)
        if score > last_score:
            clf = nb_classifier
        print("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))
        det=("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))


    def most_informative_feature_for_binary_classification(vectorizer, classifier,
                                                           n=100):  # inspect the top 30 vectors for fake and real news
        """
        See: https://stackoverflow.com/a/26980472

        Identify most important features if given a vectorizer and binary classifier. Set n to the number
        of weighted features you would like to show. (Note: current implementation merely prints and does not
        return top classes.)
        """

        class_labels = classifier.classes_
        feature_names = vectorizer.get_feature_names()  # Array mapping from feature integer indices to feature name
        topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
        topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

        for coef, feat in topn_class1:
            print(class_labels[0], coef, feat)
            fakenews = (class_labels[0], coef, feat)


        for coef, feat in reversed(topn_class2):
            print(class_labels[1], coef, feat)
            realnews = (class_labels[1], coef, feat)



            FakeRealModel.objects.create(realnews=realnews,alpha=det)
    most_informative_feature_for_binary_classification(tfidf_vectorizer, clf, n=10)
    feature_names = tfidf_vectorizer.get_feature_names()

    obj1 = FakeRealModel.objects.filter(Q(id='1') | Q(id='2') | Q(id='3') | Q(id='4') | Q(id='5') | Q(id='6') | Q(id='7') | Q(id='8') | Q(id='9') | Q(id='10'))

    return render(request, 'client/analysis3.html',{'v':obj1})



def analysis31(request):


    df = pd.read_csv('Detecting-Fake-News-with-Scikit-Learn-master/data/fake_or_real_news.csv')

    # Inspect shape of `df`
    df.shape

    # Print first lines of `df`
    df.head()

    # Set index

    # Print first lines of `df`
    df.head()

    # Set `y`
    y = df.label

    # Drop the `label` column
    df.drop("label", axis=1)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

    # Initialize the `count_vectorizer`
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(X_train)

    # Transform the test set
    count_test = count_vectorizer.transform(X_test)

    # Initialize the `tfidf_vectorizer`
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Fit and transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test set
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Get the feature names of `tfidf_vectorizer`
    print(tfidf_vectorizer.get_feature_names()[-10:])

    # Get the feature names of `count_vectorizer`
    print(count_vectorizer.get_feature_names()[:10])

    count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

    difference = set(count_df.columns) - set(tfidf_df.columns)


    set()

    print(count_df.equals(tfidf_df))

    count_df.head()

    tfidf_df.head()

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        See full source and example:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        clf = MultinomialNB()

        clf.fit(tfidf_train, y_train)
        pred = clf.predict(tfidf_test)
        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)
        cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
        plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

    clf = MultinomialNB(alpha=0.1)

    last_score = 0
    for alpha in np.arange(0, 1, .1):
        nb_classifier = MultinomialNB(alpha=alpha)
        nb_classifier.fit(tfidf_train, y_train)
        pred = nb_classifier.predict(tfidf_test)
        score = metrics.accuracy_score(y_test, pred)
        if score > last_score:
            clf = nb_classifier
        print("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))
        det=("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))


    def most_informative_feature_for_binary_classification(vectorizer, classifier,
                                                           n=100):  # inspect the top 30 vectors for fake and real news
        """
        See: https://stackoverflow.com/a/26980472

        Identify most important features if given a vectorizer and binary classifier. Set n to the number
        of weighted features you would like to show. (Note: current implementation merely prints and does not
        return top classes.)
        """

        class_labels = classifier.classes_
        feature_names = vectorizer.get_feature_names()  # Array mapping from feature integer indices to feature name
        topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
        topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

        for coef, feat in topn_class1:
            print(class_labels[0], coef, feat)
            fakenews = (class_labels[0], coef, feat)

            FakeRealModel.objects.create(fakenews=fakenews, alpha=det)
        for coef, feat in reversed(topn_class2):
            print(class_labels[1], coef, feat)
            realnews = (class_labels[1], coef, feat)




    most_informative_feature_for_binary_classification(tfidf_vectorizer, clf, n=10)
    feature_names = tfidf_vectorizer.get_feature_names()

    obj1 = FakeRealModel.objects.filter(Q(id='11') | Q(id='12') | Q(id='13') | Q(id='14') | Q(id='15') | Q(id='16') | Q(id='17') | Q(id='18') | Q(id='19') | Q(id='20'))

    return render(request, 'client/analysis31.html',{'v':obj1})




def analysis32(request):


    df = pd.read_csv('Detecting-Fake-News-with-Scikit-Learn-master/data/fake_or_real_news.csv')

    # Inspect shape of `df`
    df.shape

    # Print first lines of `df`
    df.head()

    # Set index

    # Print first lines of `df`
    df.head()

    # Set `y`
    y = df.label

    # Drop the `label` column
    df.drop("label", axis=1)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

    # Initialize the `count_vectorizer`
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(X_train)

    # Transform the test set
    count_test = count_vectorizer.transform(X_test)

    # Initialize the `tfidf_vectorizer`
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Fit and transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test set
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Get the feature names of `tfidf_vectorizer`
    print(tfidf_vectorizer.get_feature_names()[-10:])

    # Get the feature names of `count_vectorizer`
    print(count_vectorizer.get_feature_names()[:10])

    count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

    difference = set(count_df.columns) - set(tfidf_df.columns)


    set()

    print(count_df.equals(tfidf_df))

    count_df.head()

    tfidf_df.head()

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        See full source and example:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        clf = MultinomialNB()

        clf.fit(tfidf_train, y_train)
        pred = clf.predict(tfidf_test)
        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)
        cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
        plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

    clf = MultinomialNB(alpha=0.1)

    last_score = 0
    for alpha in np.arange(0, 1, .1):
        nb_classifier = MultinomialNB(alpha=alpha)
        nb_classifier.fit(tfidf_train, y_train)
        pred = nb_classifier.predict(tfidf_test)
        score = metrics.accuracy_score(y_test, pred)
        if score > last_score:
            clf = nb_classifier
        print("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))
        det=("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))

        FakeRealModel.objects.create(alpha=det)

    def most_informative_feature_for_binary_classification(vectorizer, classifier,
                                                           n=100):  # inspect the top 30 vectors for fake and real news
        """
        See: https://stackoverflow.com/a/26980472

        Identify most important features if given a vectorizer and binary classifier. Set n to the number
        of weighted features you would like to show. (Note: current implementation merely prints and does not
        return top classes.)
        """

        class_labels = classifier.classes_
        feature_names = vectorizer.get_feature_names()  # Array mapping from feature integer indices to feature name
        topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
        topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

        for coef, feat in topn_class1:
            print(class_labels[0], coef, feat)
            fakenews = (class_labels[0], coef, feat)


        for coef, feat in reversed(topn_class2):
            print(class_labels[1], coef, feat)
            realnews = (class_labels[1], coef, feat)




    most_informative_feature_for_binary_classification(tfidf_vectorizer, clf, n=10)
    feature_names = tfidf_vectorizer.get_feature_names()

    obj1 = FakeRealModel.objects.filter(Q(id='31') | Q(id='32') | Q(id='33') | Q(id='34') | Q(id='35') | Q(id='36') | Q(id='37') | Q(id='38') | Q(id='39') | Q(id='40'))

    return render(request, 'client/analysis32.html',{'v':obj1})





def analysis4(request):
    df = pd.read_csv('Detecting-Fake-News-with-Scikit-Learn-master/data/fake_or_real_news.csv')

    # Inspect shape of `df`
    df.shape

    # Print first lines of `df`
    df.head()

    # Set index

    # Print first lines of `df`
    df.head()

    # Set `y`
    y = df.label

    # Drop the `label` column
    df.drop("label", axis=1)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

    # Initialize the `count_vectorizer`
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(X_train)

    # Transform the test set
    count_test = count_vectorizer.transform(X_test)

    # Initialize the `tfidf_vectorizer`
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Fit and transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test set
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Get the feature names of `tfidf_vectorizer`
    print(tfidf_vectorizer.get_feature_names()[-10:])

    # Get the feature names of `count_vectorizer`
    print(count_vectorizer.get_feature_names()[:10])

    count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

    difference = set(count_df.columns) - set(tfidf_df.columns)

    set()

    print(count_df.equals(tfidf_df))

    count_df.head()

    tfidf_df.head()

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        See full source and example:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()



    hash_vectorizer = HashingVectorizer(stop_words='english', non_negative=True)
    hash_train = hash_vectorizer.fit_transform(X_train)
    hash_test = hash_vectorizer.transform(X_test)

    clf = MultinomialNB(alpha=.01)

    clf.fit(hash_train, y_train)
    pred = clf.predict(hash_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

    return render(request, 'client/analysis4.html')

def analysis5(request):
    df = pd.read_csv('Detecting-Fake-News-with-Scikit-Learn-master/data/fake_or_real_news.csv')

    # Inspect shape of `df`
    df.shape

    # Print first lines of `df`
    df.head()

    # Set index

    # Print first lines of `df`
    df.head()

    # Set `y`
    y = df.label

    # Drop the `label` column
    df.drop("label", axis=1)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

    # Initialize the `count_vectorizer`
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(X_train)

    # Transform the test set
    count_test = count_vectorizer.transform(X_test)

    # Initialize the `tfidf_vectorizer`
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Fit and transform the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test set
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Get the feature names of `tfidf_vectorizer`
    print(tfidf_vectorizer.get_feature_names()[-10:])

    # Get the feature names of `count_vectorizer`
    print(count_vectorizer.get_feature_names()[:10])

    count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

    difference = set(count_df.columns) - set(tfidf_df.columns)

    set()

    print(count_df.equals(tfidf_df))

    count_df.head()

    tfidf_df.head()

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        See full source and example:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    clf = LinearSVC()

    clf.fit(tfidf_train, y_train)
    pred = clf.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    accuracy = ("accuracy:   %0.3f" % score)

    cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])




    return render(request, 'client/analysis5.html',{'v':accuracy})


def uploadpage(request):
    if request.method == "POST":
        newsid =request.POST.get('newsid')
        title = request.POST.get('title')
        text = request.POST.get('text')
        label = request.POST.get('label')

        AdminModel.objects.create(newsid=newsid, title=title, text=text, label=label)

    return render(request,"client/uploadpage.html")

def view_uploadnews(request):
    obj1 = AdminModel.objects.all()
    return render(request,'client/view_uploadnews.html',{'obj1':obj1})
