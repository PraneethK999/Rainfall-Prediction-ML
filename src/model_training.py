from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_models(X, y, preprocessor):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Random Forest Pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    rf_pipeline.fit(X_train, y_train)

    rf_predictions = rf_pipeline.predict(X_test)

    rf_accuracy = accuracy_score(y_test, rf_predictions)

    # Logistic Regression Pipeline
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    lr_pipeline.fit(X_train, y_train)

    lr_predictions = lr_pipeline.predict(X_test)

    lr_accuracy = accuracy_score(y_test, lr_predictions)

    return rf_pipeline, lr_pipeline, rf_accuracy, lr_accuracy