import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, dataset_name, ch, lm):
        self.dataset_name = dataset_name
        self.ch = ch
        self.lm = lm



    def build_multilingual_embeddings(self, df, text_col: str = "content", output_path: str = "embeddings.npy", model_name: str = "intfloat/multilingual-e5-large", batch_size: int = 32):
        texts = df[text_col].fillna("").astype(str).tolist()

        # E5 requires prefix
        texts = [f"passage: {t}" for t in texts]

        model = SentenceTransformer(model_name)

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # IMPORTANT for cosine similarity
        ).astype(np.float32)

        self.ch.save_object(embeddings, output_path)
