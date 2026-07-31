import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset you built in step 2
df = pd.read_csv("quantumbridge_data/master_dataset.csv")

# X = input (what we predict FROM), y = output (what we predict)
X = df[["run_number"]]   # must be 2D — hence double brackets
y = df["error_rate"]

# Split: 15 runs to TRAIN the model, 5 runs to TEST if it actually learned anything
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"Training on {len(X_train)} runs, testing on {len(X_test)} runs")

# Create and train the model — this is the "learning" step
model = LinearRegression()
model.fit(X_train, y_train)

# What did it learn?
print(f"\nModel learned: error_rate = {model.coef_[0]:.4f} * run_number + {model.intercept_:.4f}")

# Test it on unseen data
predictions = model.predict(X_test)

print("\nPredictions vs Reality on test runs:")
for run, actual, predicted in zip(X_test["run_number"], y_test, predictions):
    print(f"  Run {run}: actual={actual:.2f}%, predicted={predicted:.2f}%, error={abs(actual-predicted):.2f}")

# R² score: how much of the variation does the model explain? (0 = useless, 1 = perfect)
score = model.score(X_test, y_test)
print(f"\nR² score: {score:.3f}")
print("(Close to 0 or negative = run_number alone doesn't predict error rate well)")
print("(Close to 1 = strong predictive relationship)")
