# Handwritten Digit Recognition - Machine Learning

## Project Description

Machine learning solution for recognizing handwritten digits (0-9) using multiple classification algorithms. Compares Decision Tree (baseline) with Random Forest, SVM, and KNN to demonstrate performance improvements.

## Dataset

- **Dataset**: `images_chiffres_codees_niveau_de_gris.csv`
- **Samples**: 2,559 (80% train, 20% test)
- **Features**: 784 pixels (28×28 grayscale images)
- **Target**: Digits 0-9

## Model Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| Decision Tree | ~73.6% | Baseline - prone to overfitting |
| Random Forest | ~85-90% | Ensemble method, reduces variance |
| SVM (RBF) | ~90-95% | Best for high-dimensional data |
| KNN | ~85-90% | Instance-based learning |

## Decision Tree Limitations

1. **Overfitting**: Deep trees memorize training data, poor generalization
2. **High Dimensionality**: Struggles with 784 pixel features, misses spatial patterns
3. **Instability**: Sensitive to noise, performance degrades with image degradation
4. **Limited Interactions**: Axis-aligned splits miss non-linear pixel relationships
5. **Image Data**: Better suited for tabular data than pixel-based classification

## Why Alternative Models Perform Better

- **Random Forest**: Ensemble voting reduces overfitting, better handles high dimensions
- **SVM**: RBF kernel captures non-linear patterns, effective in 784D space
- **KNN**: Instance-based learning adapts to local patterns, no distribution assumptions

## Technologies

- Python, Pandas, NumPy
- Scikit-learn (DecisionTree, RandomForest, SVM, KNN)
- Matplotlib, Seaborn

## Usage

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Run `handwritten_digit_recognition.ipynb` to:
- Train and compare multiple models
- View performance metrics and confusion matrices
- Understand Decision Tree limitations

## Future Improvements

- Convolutional Neural Networks (CNNs) for 95%+ accuracy
- Hyperparameter tuning with GridSearchCV
- Data augmentation (rotations, translations, noise)
- Feature engineering (PCA, normalization)
