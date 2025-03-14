# **ML Project: Predicting Vomitoxin (DON) Concentration**  

## **Project Overview**  
This project aims to predict the **Vomitoxin (DON) concentration** in agricultural samples using **machine learning models**, including **CNN, Random Forest, and XGBoost**. The workflow involves **data preprocessing, handling missing values and outliers, dimensionality reduction using PCA, model selection, hyperparameter tuning, and evaluation**.

---

## **Installation & Dependencies**  

To install the required dependencies, use:  

```bash
pip install -r requirements.txt
```

Alternatively, install packages manually:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow keras xgboost
```

---

## **Project Structure**  

```
ML_Project/
│── data/                 # Folder for dataset storage
│── models/               # Folder for trained models
│── ML_Project.py         # Main script for training and evaluation
│── requirements.txt      # Required Python packages
│── README.md             # Project documentation
│── results/              # Folder to save logs, results, and visualizations
```

---

## **Preprocessing Steps & Rationale**  

1. **Handling Missing Values:**  
   - Checked and imputed missing values using appropriate statistical measures.  

2. **Outlier Removal:**  
   - Applied **IQR-based filtering** to remove extreme values.  
   - Instead of dropping, **replaced outliers with the median** for a more robust dataset.  

3. **Feature Scaling & Transformation:**  
   - Used **StandardScaler** to normalize features.  
   - Applied **log transformation** and **Box-Cox transformation** to reduce skewness.  

4. **Dimensionality Reduction (PCA):**  
   - Selected the **top 50 numerical features** based on correlation analysis.  
   - Applied **PCA** to retain maximum variance while reducing noise.  

---

## **Model Selection & Training**  

### **1. Convolutional Neural Network (CNN)**
- **Input:** 3D feature representation  
- **Layers:** Conv1D, BatchNormalization, Dropout, Dense  
- **Optimization:** Adam optimizer, MSE loss  
- **Hyperparameter Tuning:** Grid Search on learning rates, dropout, and batch size  

### **2. Random Forest Regressor**
- **Feature Selection:** PCA-transformed features  
- **Hyperparameter Optimization:** Grid Search on estimators and depth  

### **3. XGBoost Regressor**
- **Boosting-based approach for high accuracy**  
- **Tuned using Grid Search for optimal estimators and learning rate**  

---

### **Insights:**
- **PCA improved model performance** by reducing redundant features.  
- **Random Forest and XGBoost outperformed CNN**, suggesting tree-based models work better for this dataset.  
- **Replacing outliers was more effective than removing them.**  
- **Box-Cox transformation reduced skewness**, leading to better predictions.  

---

## **Next Steps for Improvement**  
✔ **Try LSTM for sequence-based learning**  
✔ **Experiment with ensemble methods (Stacking or Blending)**  
✔ **Use SHAP values to interpret feature importance**  
✔ **Further optimize CNN architecture with advanced regularization**  

---

## **How to Run the Project**  

1. Ensure dependencies are installed (`pip install -r requirements.txt`).  
2. Run the main script:  
   ```bash
   python ML_Project.py
   ```
3. The trained models and results will be saved in the `/results/` directory.  
