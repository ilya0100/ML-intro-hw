import numpy as np

def bin_search(X, y, pipeline):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    begin = 0
    end = min(X.shape)
    while begin < end:
        middle = (begin + end) // 2
        pca_cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        if pca_cv_scores.mean() < 90:
            begin = middle + 1
        else:
            end = middle
    return begin