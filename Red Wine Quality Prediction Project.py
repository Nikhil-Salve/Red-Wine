#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv"
df = pd.read_csv(url, delimiter=',')
cutoff = 7
df['quality'] = df['quality'].apply(lambda x: 1 if x >= cutoff else 0)
X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
parameters = {
    'max_depth': [3, 5, 7, 9],
    'min_samples_leaf': [1, 3, 5, 7]
}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(clf, parameters, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_clf = grid_search.best_estimator_
y_pred = clf.predict(X_test)
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Best Hyperparameters:", grid_search.best_params_)
print("AUC:", roc_auc)


# In[ ]:




