from sklearn.svm import LinearSVC

def train_model(X_train, y_train):

    model = LinearSVC()

    model.fit(
        X_train,
        y_train
    )

    return model
