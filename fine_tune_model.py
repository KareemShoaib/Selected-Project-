import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

df = pd.read_csv('csv/small/train.csv')
df['relevance'] = pd.to_numeric(df['relevance'])

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
