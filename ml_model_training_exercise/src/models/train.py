import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Import our configurations and loader
from src.config.settings import config
from src.data.loader import load_and_prepare_data

def run_evaluation_pipeline():
    print("Loading clinical data...")
    X, y = load_and_prepare_data()
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.data.test_size, 
        random_state=config.data.random_state
    )
    
    # Scale features (essential for linear models like Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize MLflow Tracking
    mlflow.set_experiment("Model_Comparison_Exercise")

    # Define our models based on your Pydantic settings
 # Define models to iterate through
    models = {
        "Random_Forest_Tuned": (
            GridSearchCV(
                estimator=RandomForestClassifier(random_state=config.data.random_state),
                param_grid={
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                cv=5,            
                scoring="accuracy",
                n_jobs=-1        
            ),
            X_train,
            X_test
        ),
        "Logistic_Regression_Tuned": (
            GridSearchCV(
                estimator=LogisticRegression(
                    random_state=config.data.random_state,
                    solver="liblinear"  # Keeps the math stable
                ),
                param_grid={
                    "C": [0.01, 0.1, 1.0, 10.0, 100.0] # Testing different regularization strengths
                },
                cv=5,
                scoring="accuracy",
                n_jobs=-1
            ),
            X_train_scaled,  # LR requires scaling
            X_test_scaled
        )
    }

    # Train and Evaluate each model
    for name, (search_model, x_tr, x_te) in models.items():
        print(f"\nRunning Grid Search for {name}...")
        
        with mlflow.start_run(run_name=name):
            # 1. Fit the Grid Search (this tests all combinations)
            search_model.fit(x_tr, y_train)
            
            # Extract the single best model found during the search
            best_model = search_model.best_estimator_
            
            # 2. Log the WINNING hyperparameters to MLflow
            mlflow.log_params(search_model.best_params_)
            mlflow.log_param("test_size", config.data.test_size)
            
            # 3. Predict and Evaluate using the best model
            preds = best_model.predict(x_te)
            acc = accuracy_score(y_test, preds)
            
            # Log tracking metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("cv_best_score", search_model.best_score_)
            
            # Save the winning model
            mlflow.sklearn.log_model(best_model, f"{name}_best_model")
            
            print(f"=== {name} ===")
            print(f"Best Parameters Found: {search_model.best_params_}")
            print(f"Test Accuracy: {acc:.4f}")
            print(classification_report(y_test, preds))
            print("-" * 40)
            
if __name__ == "__main__":
    run_evaluation_pipeline()
