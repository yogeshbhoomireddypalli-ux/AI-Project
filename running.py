import math

# ------------------------------
# Normalization Function
# ------------------------------

def min_max_normalization(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

# ------------------------------
# Regression Metrics
# ------------------------------

def mse(y_true, y_pred):
    return sum((a - p) ** 2 for a, p in zip(y_true, y_pred)) / len(y_true)

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_total = sum((y - mean_y) ** 2 for y in y_true)
    ss_residual = sum((a - p) ** 2 for a, p in zip(y_true, y_pred))
    return 1 - (ss_residual / ss_total)

# ------------------------------
# Linear Regression using Gradient Descent
# ------------------------------

def train_linear_regression(X, y, lr=0.1, epochs=100):
    w = 0
    b = 0
    n = len(X)

    for epoch in range(epochs):
        dw = 0
        db = 0

        for i in range(n):
            y_pred = w * X[i] + b
            error = y_pred - y[i]
            dw += error * X[i]
            db += error

        dw = (2/n) * dw
        db = (2/n) * db

        w -= lr * dw
        b -= lr * db

        if epoch % 10 == 0:
            current_loss = mse(y, [w*x + b for x in X])
            print(f"Epoch {epoch} | Loss: {current_loss:.4f}")

    return w, b

# ------------------------------
# MAIN
# ------------------------------

def main():

    print("===== LINEAR REGRESSION TRAINING =====")

    # Dataset
    X = [1, 2, 3, 4, 5]
    y = [3, 5, 7, 9, 11]

    # Normalize X
    X_norm = min_max_normalization(X)

    # Train model
    w, b = train_linear_regression(X_norm, y, lr=0.5, epochs=100)

    print("\nTrained Parameters:")
    print("Weight:", w)
    print("Bias:", b)

    # Predictions
    y_pred = [w*x + b for x in X_norm]

    print("\nPredictions:", y_pred)

    # Evaluation
    print("\n===== EVALUATION =====")
    print("MSE:", mse(y, y_pred))
    print("RMSE:", rmse(y, y_pred))
    print("R2 Score:", r2_score(y, y_pred))


if __name__ == "__main__":
    main()