from manim import *

class standard_rfc(Scene):
    def construct(self):
        code = '''
def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {acc:.4f}")
'''

        gcv_conflict = Code(
            code=code,
            language="python",
            background="window"
        )

        self.add(gcv_conflict)


class feature_branch_code(Scene):
    def construct(self):
        code = '''
def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [1, 2, 5]
    }
    gcv = GridSearchCV(RandomForestClassifier(random_state=42), 
                            param_grid, cv=5)
    gcv.fit(X_train, y_train)
    gcv_best = gcv.best_estimator_

    y_pred = gcv_best.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {acc:.4f}")


'''

        gcv_conflict = Code(
            code=code,
            language="python",
            background="window"
        )

        self.add(gcv_conflict)


class merge_conflict(Scene):
    def construct(self):
        code = '''
def train_random_forest(X_train, y_train, X_test, y_test):
<<<<<<< HEAD
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
=======
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [1, 2, 5]
    }
    gcv = GridSearchCV(RandomForestClassifier(random_state=42), 
                            param_grid, cv=5)
    gcv.fit(X_train, y_train)
    gcv_best = gcv.best_estimator_
    y_pred = gcv_best.predict(X_test)
>>>>>>> feature/gridsearch

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {acc:.4f}")


'''

        gcv_conflict = Code(
            code=code,
            language="python",
            background="window"
        )

        self.add(gcv_conflict)


class merge_conflict_resolved(Scene):
    def construct(self):
        code = '''
def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [1, 2, 5]
    }
    gcv = GridSearchCV(RandomForestClassifier(random_state=42), 
                            param_grid, cv=5)
    gcv.fit(X_train, y_train)
    gcv_best = gcv.best_estimator_
    y_pred = gcv_best.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {acc:.4f}")


'''

        gcv_conflict = Code(
            code=code,
            language="python",
            background="window",
        )

        self.add(gcv_conflict)