# fine_tune_model.py

import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

# Read CSV with proper encoding and convert relevance to float
df = pd.read_csv('Book2.csv', encoding='latin1')
df['relevance'] = pd.to_numeric(df['relevance'])

# Scale the relevance scores
scaler = MinMaxScaler()
df['normalized_relevance'] = scaler.fit_transform(df[['relevance']].values)

train_examples = [
    InputExample(
        texts=[search_term, product_title], 
        label=relevance
    )
    for search_term, product_title, relevance in zip(
        df['search_term'], 
        df['product_title'], 
        df['normalized_relevance']
    )
]

model = SentenceTransformer('all-MiniLM-L6-v2')

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True,
    output_path='fine_tuned_model'
)
