import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    """Loads the clinical breast cancer dataset."""
    raw_data = load_breast_cancer()
    
    # Create a DataFrame for features (X)
    X = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
    
    # Create a Series for the target variable (y)
    y = pd.Series(raw_data.target, name="target")
    
    return X, y
    