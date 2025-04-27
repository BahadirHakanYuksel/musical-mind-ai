import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import joblib
import time

# Import feature extractor from main script
from instrument_classifier import AudioFeatureExtractor, prepare_data

# Set random seed for reproducibility
np.random.seed(42)

# Dictionary of regression models to compare
def get_regression_models():
    return {
        "Logistic Regression (Multinomial)": LogisticRegression(
            multi_class='multinomial', 
            solver='lbfgs', 
            max_iter=1000, 
            random_state=42
        ),
        "Logistic Regression (OvR)": LogisticRegression(
            multi_class='ovr', 
            solver='liblinear', 
            max_iter=1000, 
            random_state=42
        ),
        "One-vs-Rest LogisticRegression": OneVsRestClassifier(
            LogisticRegression(random_state=42)
        ),
        "One-vs-One LogisticRegression": OneVsOneClassifier(
            LogisticRegression(random_state=42)
        ),
        "SGD Classifier (log loss)": SGDClassifier(
            loss='log_loss', 
            max_iter=1000, 
            random_state=42
        ),
        "Ridge Classifier": RidgeClassifier(
            alpha=1.0, 
            random_state=42
        )
    }

def compare_regression_models(X_train, y_train, X_val, y_val):
    """Compare different regression models."""
    # Get models to compare
    models = get_regression_models()
    
    # Dictionary to store results
    results = {
        'model': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'training_time': []
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model and measure time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Calculate accuracy
        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        # Store results
        results['model'].append(name)
        results['train_accuracy'].append(train_accuracy)
        results['val_accuracy'].append(val_accuracy)
        results['training_time'].append(training_time)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
    
    return pd.DataFrame(results)

def tune_best_model(X_train, y_train, X_val, y_val, results_df):
    """Tune the best performing regression model."""
    # Find the best model
    best_model_name = results_df.loc[results_df['val_accuracy'].idxmax(), 'model']
    print(f"\nTuning the best model: {best_model_name}")
    
    # Define parameter grid based on the best model
    if "Multinomial" in best_model_name:
        model = LogisticRegression(multi_class='multinomial', random_state=42)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['newton-cg', 'sag', 'saga', 'lbfgs'],
            'max_iter': [1000, 2000]
        }
    elif "OvR" in best_model_name:
        model = LogisticRegression(multi_class='ovr', random_state=42)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['newton-cg', 'sag', 'saga', 'liblinear'],
            'max_iter': [1000, 2000]
        }
    elif "One-vs-Rest" in best_model_name:
        model = OneVsRestClassifier(LogisticRegression(random_state=42))
        param_grid = {
            'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'estimator__solver': ['newton-cg', 'sag', 'saga', 'liblinear'],
            'estimator__max_iter': [1000, 2000]
        }
    elif "One-vs-One" in best_model_name:
        model = OneVsOneClassifier(LogisticRegression(random_state=42))
        param_grid = {
            'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'estimator__solver': ['newton-cg', 'sag', 'saga', 'liblinear'],
            'estimator__max_iter': [1000, 2000]
        }
    elif "SGD" in best_model_name:
        model = SGDClassifier(loss='log_loss', random_state=42)
        param_grid = {
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'max_iter': [1000, 2000, 5000]
        }
    elif "Ridge" in best_model_name:
        model = RidgeClassifier(random_state=42)
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
    else:
        print(f"No parameter grid defined for {best_model_name}")
        return None
    
    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Print results
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    val_pred = grid_search.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation Accuracy with Tuned Model: {val_accuracy:.4f}")
    
    # Save the best model
    joblib.dump(grid_search.best_estimator_, 'best_regression_model.pkl')
    print("Saved best regression model to best_regression_model.pkl")
    
    return grid_search.best_estimator_, val_accuracy

def evaluate_model(model, X_test, y_test, label_names):
    """Evaluate the model on test data."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('regression_confusion_matrix.png')
    print("Saved confusion matrix to regression_confusion_matrix.png")
    
    return test_accuracy

def plot_model_comparison(results_df):
    """Plot comparison of regression models."""
    plt.figure(figsize=(12, 10))
    
    # Model accuracy
    plt.subplot(2, 1, 1)
    bar_width = 0.35
    index = np.arange(len(results_df['model']))
    
    plt.bar(index, results_df['train_accuracy'], bar_width, label='Training Accuracy', color='blue')
    plt.bar(index + bar_width, results_df['val_accuracy'], bar_width, label='Validation Accuracy', color='green')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Regression Model Comparison: Accuracy')
    plt.xticks(index + bar_width/2, results_df['model'], rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1)
    
    # Training time
    plt.subplot(2, 1, 2)
    plt.bar(results_df['model'], results_df['training_time'], color='red')
    plt.xlabel('Model')
    plt.ylabel('Training Time (seconds)')
    plt.title('Regression Model Comparison: Training Time')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('regression_model_comparison.png')
    print("Saved model comparison plot to regression_model_comparison.png")

def main():
    # Paths to data
    train_metadata_path = 'Metadata_Train.csv'
    test_metadata_path = 'Metadata_Test.csv'
    train_dir = 'Train_submission'
    test_dir = 'Test_submission'
    
    # Prepare data
    audio_paths, y_encoded, label_encoder = prepare_data(train_metadata_path, train_dir)
    
    # Feature extraction
    feature_extractor = AudioFeatureExtractor()
    
    # Extract features
    print("Extracting features...")
    X, y = feature_extractor.extract_features_dataset(audio_paths, y_encoded)
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Compare regression models
    results_df = compare_regression_models(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Print results
    print("\nRegression Model Comparison Results:")
    print(results_df)
    
    # Plot model comparison
    plot_model_comparison(results_df)
    
    # Tune the best model
    best_model, _ = tune_best_model(X_train_scaled, y_train, X_val_scaled, y_val, results_df)
    
    # Evaluate the tuned model on test data
    evaluate_model(best_model, X_test_scaled, y_test, label_encoder.classes_)
    
    # Save results to CSV
    results_df.to_csv('regression_model_results.csv', index=False)
    print("Saved results to regression_model_results.csv")

if __name__ == "__main__":
    main() 