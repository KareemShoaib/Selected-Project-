import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr, spearmanr
import torch
from torch import nn
import matplotlib.pyplot as plt

def evaluate_model():
    # Load the data
    df = pd.read_csv('csv/small/train.csv')
    df['relevance'] = pd.to_numeric(df['relevance'])
    
    # Normalize relevance scores (same as in training)
    scaler = MinMaxScaler()
    df['normalized_relevance'] = scaler.fit_transform(df[['relevance']].values)
    
    # Split data (use 80% for evaluation since we used all data for training)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Evaluating on {len(test_df)} samples...")
    
    # Load the fine-tuned model
    model = SentenceTransformer('fine_tuned_model')
    
    # Get predictions
    search_terms = test_df['search_term'].tolist()
    product_titles = test_df['product_title'].tolist()
    true_relevance = test_df['normalized_relevance'].tolist()
    
    # Calculate cosine similarities (predictions)
    search_embeddings = model.encode(search_terms)
    product_embeddings = model.encode(product_titles)
    
    # Calculate cosine similarity between search terms and product titles
    cosine_sim = nn.functional.cosine_similarity(
        torch.tensor(search_embeddings), 
        torch.tensor(product_embeddings)
    ).numpy()
    
    # Convert cosine similarities to [0, 1] range (since they're [-1, 1])
    predictions = (cosine_sim + 1) / 2
    
    # Calculate various metrics
    mse = mean_squared_error(true_relevance, predictions)
    mae = mean_absolute_error(true_relevance, predictions)
    
    # Correlation metrics
    pearson_corr, pearson_p = pearsonr(true_relevance, predictions)
    spearman_corr, spearman_p = spearmanr(true_relevance, predictions)
    
    print("\n=== Model Evaluation Results ===")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
    
    # Classification accuracy at different thresholds
    print("\n=== Classification Accuracy ===")
    
    # Convert to original relevance scale for classification
    true_relevance_original = test_df['relevance'].tolist()
    predictions_original = predictions * (3 - 1) + 1  # Scale back to 1-3 range
    
    # Binary classification: relevant (>= 2.5) vs not relevant (< 2.5)
    true_binary = [1 if r >= 2.5 else 0 for r in true_relevance_original]
    pred_binary = [1 if p >= 2.5 else 0 for p in predictions_original]
    binary_accuracy = np.mean([t == p for t, p in zip(true_binary, pred_binary)])
    print(f"Binary Accuracy (threshold=2.5): {binary_accuracy:.4f}")
    
    # 3-class classification (Low: 1-1.5, Medium: 1.5-2.5, High: 2.5-3)
    def classify_relevance(score):
        if score < 1.5:
            return 0  # Low
        elif score < 2.5:
            return 1  # Medium
        else:
            return 2  # High
    
    true_3class = [classify_relevance(r) for r in true_relevance_original]
    pred_3class = [classify_relevance(p) for p in predictions_original]
    class3_accuracy = np.mean([t == p for t, p in zip(true_3class, pred_3class)])
    print(f"3-Class Accuracy: {class3_accuracy:.4f}")
    
    # Exact match accuracy (rounded to nearest 0.5)
    true_rounded = [round(r * 2) / 2 for r in true_relevance_original]
    pred_rounded = [round(p * 2) / 2 for p in predictions_original]
    exact_accuracy = np.mean([t == p for t, p in zip(true_rounded, pred_rounded)])
    print(f"Exact Match Accuracy (rounded to 0.5): {exact_accuracy:.4f}")
    
    # Show some examples
    print("\n=== Sample Predictions ===")
    for i in range(min(10, len(test_df))):
        idx = test_df.iloc[i].name
        print(f"Search: '{search_terms[i]}'")
        print(f"Product: '{product_titles[i]}'")
        print(f"True Relevance: {true_relevance_original[i]:.2f}")
        print(f"Predicted Relevance: {predictions_original[i]:.2f}")
        print(f"Cosine Similarity: {cosine_sim[i]:.4f}")
        print("-" * 50)
    
    # Create visualization
    plt.figure(figsize=(12, 4))
    
    # Scatter plot of predictions vs true values
    plt.subplot(1, 3, 1)
    plt.scatter(true_relevance_original, predictions_original, alpha=0.6)
    plt.plot([1, 3], [1, 3], 'r--', label='Perfect prediction')
    plt.xlabel('True Relevance')
    plt.ylabel('Predicted Relevance')
    plt.title('Predictions vs True Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribution of true values
    plt.subplot(1, 3, 2)
    plt.hist(true_relevance_original, bins=20, alpha=0.7, label='True')
    plt.hist(predictions_original, bins=20, alpha=0.7, label='Predicted')
    plt.xlabel('Relevance Score')
    plt.ylabel('Frequency')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    plt.subplot(1, 3, 3)
    errors = np.array(predictions_original) - np.array(true_relevance_original)
    plt.hist(errors, bins=20, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.axvline(x=0, color='r', linestyle='--', label='Perfect prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'binary_accuracy': binary_accuracy,
        'class3_accuracy': class3_accuracy,
        'exact_accuracy': exact_accuracy
    }

if __name__ == "__main__":
    results = evaluate_model() 