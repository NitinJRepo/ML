# Ensemble methods work best when the predictors are as independent from one another as possible. One way to get diverse classifiers is to train them using very different algorithms.
# This increases the chance that they will make very different types of errors, improving the ensemble’s accuracy.

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score

data = datasets.load_digits()

X_data = data.images   # load X_data
y_data = data.target   # load y_data

X_data = X_data.reshape(X_data.shape[0], X_data.shape[1] * X_data.shape[2])    # flatten X_data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 7)    # split data into train & test set

clf_svc = SVC() 
clf_dt = DecisionTreeClassifier()
clf_gnb = GaussianNB()

voting_clf = VotingClassifier(estimators = [('svm', clf_svc), ('decision_tree', clf_dt), ('naive_bayes', clf_gnb)], voting = 'hard')
voting_clf.fit(X_train, y_train)

for clf in (clf_svc, clf_dt, clf_gnb, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

