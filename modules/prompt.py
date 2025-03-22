"""Prompt generation and management for data processing tasks."""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from LLM4DP.modules.data_loader import DataLoader

logger = logging.getLogger(__name__)

class PromptManager:
    def __init__(self, model_name: str = "sentence-transformers/sentence-t5-base"):
        """
        Initialize PromptManager with embedding model and data loader.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = self._setup_embedding_model(model_name)
        self.data_loader = DataLoader()
        
    def _setup_embedding_model(self, model_name: str) -> SentenceTransformer:
        """Initialize sentence transformer model."""
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        return SentenceTransformer(model_name)
        
    def generate_prompt(self, data: Union[str, pd.DataFrame], 
                       method: str = 'random',
                       num_examples: int = 10,
                       task_type: str = 'entity_matching',
                       use_embeddings: bool = False) -> str:
        """
        Generate prompts using various strategies.
        
        Args:
            data: Input data or path to data
            method: Prompt generation method ('random', 'manual', 'validation')
            num_examples: Number of examples to include
            task_type: Type of task ('entity_matching', 'imputation', etc.)
            use_embeddings: Whether to use embeddings for example selection
            
        Returns:
            Generated prompt string
        """
        # Load data if path is provided
        if isinstance(data, str):
            data = self.data_loader.read_data(data)
            if isinstance(data, dict):
                data = data.get('train', next(iter(data.values())))
                
        if method == 'random':
            return self._generate_random_prompt(data, num_examples)
        elif method == 'manual':
            return self._generate_manual_prompt(data)
        elif method == 'validation':
            return self._generate_validation_prompt(data, num_examples, task_type, use_embeddings)
        else:
            raise ValueError(f"Unknown prompt generation method: {method}")
            
    def _generate_random_prompt(self, data: pd.DataFrame, num_examples: int) -> str:
        """Generate prompt using random examples."""
        examples = self.data_loader.sample_data(data, max_samples=num_examples)
        return self._format_examples(examples)
        
    def _generate_manual_prompt(self, data: pd.DataFrame) -> str:
        """Generate prompt using predefined templates."""
        # Implementation depends on your specific templates
        raise NotImplementedError("Manual prompt generation not implemented")
        
    def _generate_validation_prompt(self, data: pd.DataFrame, num_examples: int,
                                  task_type: str, use_embeddings: bool) -> str:
        """Generate prompt using validation examples."""
        if not use_embeddings:
            return self._generate_random_prompt(data, num_examples)
            
        # Find errors in predictions
        errors = data[
            data['label_str'].str.lower().str.strip() != 
            data['preds'].str.lower().str.strip()
        ]
        
        # Get embeddings and find hard samples
        embeddings = self._get_embeddings(data)
        samples = self._get_hard_samples(data, errors.index, embeddings, num_examples)
        
        # Balance classes if needed
        if task_type in {'entity_matching', 'error_detection'}:
            samples = self._balance_yes_no_classes(samples, data)
            
        return self._format_examples(samples)
        
    def _get_embeddings(self, data: pd.DataFrame, text_col: str = 'text') -> np.ndarray:
        """Get embeddings for text data."""
        logger.info("Extracting text embeddings")
        return self.model.encode(data[text_col].tolist())
        
    def _get_hard_samples(self, data: pd.DataFrame, error_indices: pd.Index,
                         embeddings: np.ndarray, num_samples: int) -> pd.DataFrame:
        """Select hard samples based on embeddings."""
        # Calculate similarity matrix
        sim_matrix = np.triu(cosine_similarity(embeddings))
        
        # Find most similar pairs with different labels
        pairs = []
        for i in range(sim_matrix.shape[0]):
            similar_indices = np.argsort(sim_matrix[i])[::-1]
            for j in similar_indices[1:num_samples + 1]:
                if i == j:
                    continue
                if ((i in error_indices or j in error_indices) and
                    data.iloc[i]['label_str'] != data.iloc[j]['label_str']):
                    pairs.extend([i, j])
                    
        return data.iloc[list(set(pairs))[:num_samples]]
        
    def _balance_yes_no_classes(self, samples: pd.DataFrame, 
                              full_data: pd.DataFrame) -> pd.DataFrame:
        """Ensure both Yes and No classes are represented."""
        yes_samples = samples[samples['label_str'].str.strip() == 'Yes\n']
        no_samples = samples[samples['label_str'].str.strip() == 'No\n']
        
        if len(yes_samples) == 0:
            # Add a Yes example
            yes_example = full_data[full_data['label_str'].str.strip() == 'Yes\n'].sample(1)
            samples = pd.concat([samples.iloc[:-1], yes_example])
        elif len(no_samples) == 0:
            # Add a No example
            no_example = full_data[full_data['label_str'].str.strip() == 'No\n'].sample(1)
            samples = pd.concat([samples.iloc[:-1], no_example])
            
        return samples
        
    def _format_examples(self, examples: pd.DataFrame) -> str:
        """Format examples into prompt string."""
        formatted = []
        for _, row in examples.iterrows():
            text = row['text'].strip()
            label = row['label_str'].strip()
            formatted.append(f"{text} {label}")
            
        return "\n\n".join(formatted) + "\n"