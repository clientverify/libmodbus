#!/usr/bin/env python

from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import sys

# >>> from sklearn import tree
# >>> X = [[0, 0], [1, 1]]
# >>> Y = [0, 1]
# >>> clf = tree.DecisionTreeClassifier()
# >>> clf = clf.fit(X, Y)

def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0]*(target_len - len(some_list))

TARGET_LEN = 500

# read input
raw_data = []
raw_lengths = []
for line in sys.stdin:
    fields = line.strip().strip("[").strip("]").split("][")
    intfields = [int(x,16) for x in fields]
    raw_lengths.append(len(intfields))
    raw_data.append(pad_or_truncate(intfields, TARGET_LEN))

X = raw_data
Y = raw_lengths

print("Loaded %d samples ranging from %d-%d bytes" %
      (len(raw_data), min(raw_lengths), max(raw_lengths)))

clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf, X, Y, cv=5)
print("Decision Tree Classifier scores:", scores)

# clf.fit(X,Y)
# print "Done"

# Yprime = clf.predict(X)
# print Yprime
# for i in xrange(len(Yprime)):
#     if (Y[i] == Yprime[i]):
#         status = "CORRECT"
#     else:
#         status = "WRONG"
#     print(Y[i], Yprime[i], status)

adaboost_clf = AdaBoostClassifier(n_estimators=10)
scores = cross_val_score(adaboost_clf, X, Y, cv=5)
print("AdaBoost Classifier scores:", scores)
