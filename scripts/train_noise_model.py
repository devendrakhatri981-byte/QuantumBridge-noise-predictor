"""
QuantumBridge — Noise Prediction Model Training

Trains a linear regression model to predict quantum circuit error rate
from circuit structure (CNOT gate count) and backend identity.

This is QuantumBridge's core validated model (see docs/research_log.pdf,
Entry 007), achieving R2=0.747 on held-out real IBM Quantum hardware data.

Usage:
    python train_noise_model.py --data data/full_project_dataset.csv
"""

import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


def train_model(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Train the QuantumBridge noise prediction model.

    Parameters
    ----------
    data_path : str
        Path to the unified dataset CSV. Must contain columns:
        cnot_count, backend, error_rate
    test_size : float
        Fraction of data held out for evaluation
    random_state : int
        Seed for reproducible train/test split

    Returns
    -------
    model : LinearRegression
        The trained model
    metrics : dict
        R2 and MAE on the held-out test set
    """
    df = pd.read_csv(data_path)
    df["is_kingston"] = (df["backend"] == "ibm_kingston").astype(int)

    X = df[["cnot_count", "is_kingston"]]
    y = df["error_rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "intercept": model.intercept_,
        "cnot_weight": model.coef_[0],
        "kingston_weight": model.coef_[1],
    }

    return model, metrics


def predict_error_rate(model, cnot_count: int, is_kingston: bool = False) -> float:
    """
    Predict the expected error rate for a new circuit.

    Parameters
    ----------
    model : LinearRegression
        A model trained by train_model()
    cnot_count : int
        Number of CNOT (two-qubit entangling) gates in the circuit
    is_kingston : bool
        Whether the target backend is ibm_kingston (False = ibm_fez)

    Returns
    -------
    float
        Predicted error rate as a percentage
    """
    import numpy as np
    X_new = np.array([[cnot_count, int(is_kingston)]])
    return float(model.predict(X_new)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the QuantumBridge noise model")
    parser.add_argument("--data", default="data/full_project_dataset.csv",
                         help="Path to unified dataset CSV")
    args = parser.parse_args()

    model, metrics = train_model(args.data)

    print("QuantumBridge Noise Model — Training Complete")
    print("-" * 50)
    print(f"error_rate = {metrics['intercept']:.3f} "
          f"+ ({metrics['cnot_weight']:.3f} x cnot_count) "
          f"+ ({metrics['kingston_weight']:.3f} x is_kingston)")
    print(f"\nTest R2:  {metrics['r2']:.3f}")
    print(f"Test MAE: {metrics['mae']:.2f} percentage points")

    print("\nExample predictions:")
    for cnots in [1, 2, 3, 4]:
        pred_fez = predict_error_rate(model, cnots, is_kingston=False)
        pred_kingston = predict_error_rate(model, cnots, is_kingston=True)
        print(f"  {cnots} CNOTs -> ibm_fez: {pred_fez:.2f}%,  ibm_kingston: {pred_kingston:.2f}%")
