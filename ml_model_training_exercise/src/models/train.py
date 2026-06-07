import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

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
    models = {
        "Random_Forest": (
            RandomForestClassifier(
                n_estimators=config.rf.n_estimators,
                max_depth=config.rf.max_depth,
                random_state=config.data.random_state
            ),
            config.rf.model_dump(),
            X_train,  # RF doesn't strictly need scaling
            X_test
        ),
        "Logistic_Regression": (
            LogisticRegression(
                C=config.lr.C,
                max_iter=config.lr.max_iter,
                random_state=config.data.random_state,
                solver="liblinear"
            ),
            config.lr.model_dump(),
            X_train_scaled,  # LR requires scaling
            X_test_scaled
        )
    }

    # Train and Evaluate each model
    for name, (model, params, x_tr, x_te) in models.items():
        print(f"\nTraining {name}...")
        
        with mlflow.start_run(run_name=name):
            # Log hyperparameters to MLflow
            mlflow.log_params(params)
            mlflow.log_param("test_size", config.data.test_size)
            
            # Train the model
            model.fit(x_tr, y_train)
            
            # Predict and Evaluate
            preds = model.predict(x_te)
            acc = accuracy_score(y_test, preds)
            
            # Log tracking metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, f"{name}_model")
            
            print(f"=== {name} ===")
            print(f"Accuracy: {acc:.4f}")
            print(classification_report(y_test, preds))
            print("-" * 40)

if __name__ == "__main__":
    run_evaluation_pipeline()
