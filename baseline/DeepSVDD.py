# unsupervised methods
from deepod.models.tabular import DeepSVDD
clf = DeepSVDD()
clf.fit(X_train, y=None)
scores = clf.decision_function(X_test)

# weakly-supervised methods
from deepod.models.tabular import DevNet
clf = DevNet()
clf.fit(X_train, y=semi_y) # semi_y uses 1 for known anomalies, and 0 for unlabeled data
scores = clf.decision_function(X_test)

# evaluation of tabular anomaly detection
from deepod.metrics import tabular_metrics
auc, ap, f1 = tabular_metrics(y_test, scores)