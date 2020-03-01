from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score

data = datasets.load_digits()

X_data = data.images   # load X_data
y_data = data.target   # load y_data

X_data = X_data.reshape(X_data.shape[0], X_data.shape[1] * X_data.shape[2])    # flatten X_data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 7)    # split data into train & test set

rf_classifier = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=5)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
print(accuracy_score(y_pred, y_test))
