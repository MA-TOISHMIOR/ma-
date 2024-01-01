import pandas as pd
from sklearn.preprocessing import LabelEncoder

titanic = pd.read_csv("train.csv")

print("数据列表：")
print(titanic)

print("\n数据信息：")
print(titanic.info())

print("\n数据描述：")
print(titanic.describe())

titanic = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
titanic = titanic.dropna()
titanic['Survived'] = titanic['Survived'].astype('category')

label_encoder = LabelEncoder()
titanic['Sex'] = label_encoder.fit_transform(titanic['Sex'])
titanic['Embarked'] = label_encoder.fit_transform(titanic['Embarked'])
print(titanic)


import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

titanic = pd.read_csv("train.csv")

titanic = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
titanic = titanic.dropna()
titanic['Survived'] = titanic['Survived'].astype('category')

label_encoder = LabelEncoder()
titanic['Sex'] = label_encoder.fit_transform(titanic['Sex'])
titanic['Embarked'] = label_encoder.fit_transform(titanic['Embarked'])


titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1
titanic['IsAlone'] = titanic['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
titanic['FarePerPerson'] = titanic['Fare'] / titanic['FamilySize']

scaler = StandardScaler()
titanic[['Age', 'Fare', 'FamilySize', 'FarePerPerson']] = scaler.fit_transform(titanic[['Age', 'Fare', 'FamilySize', 'FarePerPerson']])

X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC()
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(svm_model, param_grid, scoring='accuracy', cv=5)
svm_grid.fit(X_train, y_train)

svm_model = svm_grid.best_estimator_
svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

knn_model = KNeighborsClassifier()
param_grid = {'n_neighbors': [3, 5, 7, 9]}
knn_grid = GridSearchCV(knn_model, param_grid, scoring='accuracy', cv=5)
knn_grid.fit(X_train, y_train)

knn_model = knn_grid.best_estimator_
knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)

svm_precision = precision_score(y_test, svm_predictions)
knn_precision = precision_score(y_test, knn_predictions)

svm_recall = recall_score(y_test, svm_predictions)
knn_recall = recall_score(y_test, knn_predictions)

svm_f1_score = f1_score(y_test, svm_predictions)
knn_f1_score = f1_score(y_test, knn_predictions)

svm_auc_roc = roc_auc_score(y_test, svm_predictions)
knn_auc_roc = roc_auc_score(y_test, knn_predictions)

print("SVM accuracy:", svm_accuracy)
print("KNN accuracy:", knn_accuracy)

print("SVM precision:", svm_precision)
print("KNN precision:", knn_precision)

print("SVM recall:", svm_recall)
print("KNN recall:", knn_recall)

print("SVM F1 score:", svm_f1_score)
print("KNN F1 score:", knn_f1_score)

print("SVM AUC-ROC:", svm_auc_roc)
print("KNN AUC-ROC:", knn_auc_roc)

labels = ['SVM', 'KNN']
accuracies = [svm_accuracy, knn_accuracy]

import matplotlib.pyplot as plt
plt.bar(labels, accuracies)
plt.title("Accuracy comparison")
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.show()


models = ['SVM', 'KNN']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']

svm_scores = [svm_accuracy, svm_precision, svm_recall, svm_f1_score, svm_auc_roc]
knn_scores = [knn_accuracy, knn_precision, knn_recall, knn_f1_score, knn_auc_roc]

x = range(len(metrics))
width = 0.35

plt.bar(x, svm_scores, width, label='SVM')
plt.bar([i + width for i in x], knn_scores, width, label='KNN')

plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Evaluation Results')
plt.xticks([i + width/2 for i in x], metrics)
plt.legend()

plt.show()







