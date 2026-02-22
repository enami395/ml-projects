# Medical Costs Prediction - Linear Regression

## Project Description

This project uses supervised machine learning (linear regression) to predict medical charges based on patient characteristics. The goal is to build a regression model capable of estimating healthcare costs for individuals.

## Dataset

- **Dataset**: `frais_medicaux.txt`
- **Number of samples**: 1,338
- **Features**: 
  - `age`: Age of the patient
  - `sex`: Gender (male/female)
  - `bmi`: Body Mass Index
  - `children`: Number of children/dependents
  - `smoker`: Smoking status (yes/no)
  - `region`: Region of residence (northeast/northwest/southeast/southwest)
- **Target variable**: `charges` (medical costs in dollars)

## Methodology

### 1. Exploratory Data Analysis (EDA)

- Data overview and descriptive statistics
- Distribution analysis of the target variable
- Visualization of relationships between features and charges
- Correlation analysis between variables
- Key findings:
  - Smokers have significantly higher medical charges than non-smokers
  - Age shows a positive correlation with charges
  - BMI shows a weak to moderate correlation
  - Balanced distribution for sex and regions
  - Imbalanced distribution for smokers (20% of sample)

### 2. Data Preprocessing

- **Categorical encoding**:
  - Binary encoding for sex (female=0, male=1) and smoker (no=0, yes=1)
  - One-hot encoding for region (4 nominal categories)
- **Feature selection**: All features retained for the model
- **Train/test split**: 80/20 split with random_state=42

### 3. Model Training

- **Algorithm**: Linear Regression
- **Features used**: age, sex, bmi, children, smoker, region (one-hot encoded)
- **Total features**: 9 (after one-hot encoding)

### 4. Model Evaluation

The model is evaluated using multiple metrics:
- **MSE** (Mean Squared Error): Penalizes large errors
- **RMSE** (Root Mean Squared Error): Same unit as target, more interpretable
- **MAE** (Mean Absolute Error): Average error magnitude
- **R²** (Coefficient of Determination): Proportion of variance explained

## Results

### Model Performance

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| MSE | 37,277,681.70 | 33,596,915.85 |
| RMSE | 6,105.55 | 5,796.28 |
| MAE | 4,208.23 | 4,181.19 |
| R² | 0.7417 | 0.7836 |

### Key Findings

- **R² Score**: 0.7417 (train) / 0.7836 (test)
  - The model explains 74% of variance on training and 78% on test
  - Higher R² on test indicates good generalization without overfitting
  - Decent performance with room for improvement

### Feature Importance

The model coefficients reveal the importance of each feature:

1. **Smoker** (23,651.13): By far the most important feature, confirming major impact on charges
2. **Region (northeast)** (459.59): Moderate positive impact
3. **Children** (425.28): Moderate positive impact
4. **BMI** (337.09): Moderate positive impact
5. **Age** (256.98): Moderate positive impact
6. **Region (southwest)** (-350.21): Moderate negative impact
7. **Region (southeast)** (-198.28): Weak negative impact
8. **Region (northwest)** (88.91): Weak positive impact
9. **Sex** (-18.59): Weak negative impact

### Interpretation

- Smoking status is the strongest predictor of medical costs
- Age, BMI, and number of children have moderate positive impacts
- Sex has minimal impact on charges
- Regional differences exist but are relatively small compared to smoking status

## Technologies Used

- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning (LinearRegression, train_test_split, metrics)
- **Matplotlib/Seaborn**: Data visualization

## Usage

1. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

2. Run the notebook `medical_costs_prediction.ipynb` to:
   - Load and explore the data
   - Preprocess categorical variables
   - Train the linear regression model
   - Evaluate model performance
   - Visualize predictions vs actual values
   - Analyze feature importance

3. Make predictions:
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the trained model (if saved)
# model = joblib.load('linear_regression_model.pkl')

# Example prediction
new_patient = pd.DataFrame({
    'age': [45],
    'sex': [1],  # 1 for male, 0 for female
    'bmi': [28.5],
    'children': [2],
    'smoker': [0],  # 0 for no, 1 for yes
    'region_northeast': [False],
    'region_northwest': [True],
    'region_southeast': [False],
    'region_southwest': [False]
})

# prediction = model.predict(new_patient)
```

## Conclusion

The linear regression model successfully predicts medical costs with an R² score of 0.78 on the test set. The model identifies smoking status as the most critical factor affecting medical charges, which aligns with medical knowledge. The model demonstrates good generalization without overfitting.

**Potential Improvements**:
- Feature engineering (e.g., BMI categories, age groups)
- Polynomial features for non-linear relationships
- Regularization techniques (Ridge, Lasso)
- Other regression algorithms (Random Forest, Gradient Boosting)

**Note**: This model is for educational purposes and should not be used for actual medical cost predictions without proper validation and medical expertise.
