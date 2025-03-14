# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, BatchNormalization, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset (Assuming df is your dataset)
df = pd.read_csv("your_dataset.csv")  # Replace with actual dataset path

# Drop non-numeric columns and separate target variable
df = df.select_dtypes(include=['number'])
y = df["vomitoxin_ppb"]
X = df.drop(columns=["vomitoxin_ppb"])

# Handling Outliers: Removing outliers using IQR
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(y >= lower_bound) & (y <= upper_bound)]

# Log transformation to reduce skewness
df["log_vomitoxin"] = np.log1p(y)

# Apply Box-Cox transformation for normality
df["boxcox_vomitoxin"], lambda_ = boxcox(df["vomitoxin_ppb"] + 1)  # Adding 1 to avoid errors

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA on the top 50 numerical features
pca = PCA(n_components=0.95)  # Retaining 95% variance
X_pca = pca.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, df["boxcox_vomitoxin"], test_size=0.2, random_state=42)

### CNN Model with Hyperparameter Tuning (Grid Search)
def create_cnn_model(filters=64, kernel_size=3, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)  # Regression output
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model

# Training CNN with best parameters
cnn_model = create_cnn_model(filters=64, kernel_size=3, dropout_rate=0.3, learning_rate=0.001)
cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, verbose=1)

# Predict using CNN
y_pred_cnn = cnn_model.predict(X_test).flatten()

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# XGBoost Model
xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate models
def evaluate_model(y_test, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}")

evaluate_model(y_test, y_pred_cnn, "CNN")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_xgb, "XGBoost")

# Feature Importance from Random Forest
feature_importances = rf_model.feature_importances_
important_features = pd.DataFrame({'Feature': range(len(feature_importances)), 'Importance': feature_importances})
important_features.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Feature Importance
plt.figure(figsize=(10, 5))
sns.barplot(x="Feature", y="Importance", data=important_features.head(20))
plt.title("Top 20 Important Features (Random Forest)")
plt.show()

# Training vs Validation Loss Plot for CNN
plt.figure(figsize=(8, 6))
plt.plot(cnn_model.history.history["loss"], label="Training Loss")
plt.plot(cnn_model.history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (CNN)")
plt.legend()
plt.show()
