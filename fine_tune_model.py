import pandas as pd
import numpy as np
import re
import string
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def clean_text(self, text):
        """Comprehensive text cleaning for product search"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove special characters but keep important ones for products
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Handle measurements and units (keep them meaningful)
        text = re.sub(r'(\d+)\s*-\s*(\w+)', r'\1\2', text)  # "12-gauge" -> "12gauge"
        text = re.sub(r'(\d+)\s*(\w+)', r'\1\2', text)      # "1 gal" -> "1gal"
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, text):
        """Extract meaningful features from product text"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        features = []
        
        # Extract brand information (first few words often contain brand)
        words = text.split()
        if len(words) > 0:
            potential_brand = words[0]
            features.append(f"brand_{potential_brand}")
        
        # Extract measurements
        measurements = re.findall(r'\d+(?:\.\d+)?(?:inch|in|ft|gal|gauge|lb)', text)
        features.extend([f"measure_{m}" for m in measurements])
        
        colors = re.findall(r'\b(?:black|white|brown|gray|grey|red|blue|green|yellow|silver|gold)\b', text)
        features.extend([f"color_{c}" for c in colors])
        
        return " ".join(features)
    
    def remove_stopwords_and_stem(self, text):
        """Remove stopwords and apply stemming"""
        if pd.isna(text):
            return ""
        
        words = text.split()
        important_words = {'with', 'for', 'in', 'on', 'over', 'under'}
        filtered_words = []
        
        for word in words:
            if word not in self.stop_words or word in important_words:
                stemmed = self.stemmer.stem(word)
                filtered_words.append(stemmed)
        
        return " ".join(filtered_words)


if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

    df = pd.read_csv('csv/small/train.csv')

    df = df.drop_duplicates()
    df = df.dropna(subset=['product_title', 'search_term', 'relevance'])
    print(f"After removing duplicates and nulls: {df.shape}")

    preprocessor = Preprocessor()
    df['search_term_clean'] = df['search_term'].apply(preprocessor.clean_text)
    df['product_title_clean'] = df['product_title'].apply(preprocessor.clean_text)

    df['search_term_features'] = df['search_term'].apply(preprocessor.extract_features)
    df['product_title_features'] = df['product_title'].apply(preprocessor.extract_features)

    df['search_term'] = df['search_term_clean'] + " " + df['search_term_features']
    df['product_title'] = df['product_title_clean'] + " " + df['product_title_features']
    df = df.drop(columns=['search_term_clean', 'product_title_clean', 'search_term_features', 'product_title_features'])

    df['search_term'] = df['search_term'].apply(preprocessor.remove_stopwords_and_stem)
    df['product_title'] = df['product_title'].apply(preprocessor.remove_stopwords_and_stem)

    print(f"Final dataset shape: {df.shape}")
    print(df.head())

    Q1 = df['relevance'].quantile(0.25)
    Q3 = df['relevance'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    outliers_count = len(df[(df['relevance'] < lower_bound) | (df['relevance'] > upper_bound)])
    print(f"Number of outliers: {outliers_count}")

    df['relevance'] = df['relevance'].clip(lower_bound, upper_bound)

    scaler = MinMaxScaler()
    df['relevance'] = scaler.fit_transform(df[['relevance']])

    print("\nProcessed relevance distribution:")
    print(df['relevance'].describe())
    print(df.head())

    train_examples = [
        InputExample(
            texts=[search_term, product_title], 
            label=float(relevance)
        )
        for search_term, product_title, relevance in zip(
            df_final['search_term_enhanced'], 
            df_final['product_title_enhanced'], 
            df_final['relevance_normalized']
        )
    ]

    print(f"Created {len(train_examples)} training examples")

    train_df, val_df = train_test_split(df_final, test_size=0.2, random_state=42)

    train_examples = [
        InputExample(texts=[s, p], label=float(r))
        for s, p, r in zip(train_df['search_term_enhanced'], train_df['product_title_enhanced'], train_df['relevance_normalized'])
    ]

    val_examples = [
        InputExample(texts=[s, p], label=float(r))
        for s, p, r in zip(val_df['search_term_enhanced'], val_df['product_title_enhanced'], val_df['relevance_normalized'])
    ]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=16)

    val_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='val')

    model = SentenceTransformer('all-MiniLM-L6-v2')
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=val_evaluator,
        epochs=3,
        warmup_steps=100,
        evaluation_steps=100,
        show_progress_bar=True,
        output_path='fine_tuned_model'
    )
    
    print("Fine-tuning completed! Model saved to 'fine_tuned_model' directory.")
