from sklearn.model_selection import KFold


def get_kfold_split(files_train, k=5):
    kfold_split = dict()
    skf = KFold(n_splits=k,shuffle=True,random_state=42)
    for fold,(train_idx,val_idx) in enumerate(skf.split(files_train)):
        kfold_split[fold] = dict(
            train = files_train[train_idx],
            validation = files_train[val_idx]
        )
    return kfold_split
