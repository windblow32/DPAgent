"""Data loading and serialization utilities."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, sep_tok: str = ".", nan_tok: str = "nan", random_state: int = 42):
        """
        Initialize DataLoader with custom separators and tokens.
        
        Args:
            sep_tok: Separator token for serialization
            nan_tok: Token to represent NaN values
            random_state: Random seed for reproducibility
        """
        self.sep_tok = sep_tok
        self.nan_tok = nan_tok
        self.random_state = random_state
        
    def read_data(self, data_dir: str, class_balanced: bool = False,
                  add_instruction: bool = False, task_instruction_idx: int = 0,
                  max_train_samples: int = -1, max_train_percent: float = -1,
                  shuffle_data: bool = True) -> Dict:
        """
        Read data from directory with support for various data processing tasks.
        
        Args:
            data_dir: Directory containing the data
            class_balanced: Whether to balance classes in training data
            add_instruction: Whether to add task-specific instructions
            task_instruction_idx: Index of task instruction to use
            max_train_samples: Maximum number of training samples (-1 for no limit)
            max_train_percent: Maximum percentage of training data to use (-1 for no limit)
            shuffle_data: Whether to shuffle the data
            
        Returns:
            Dictionary containing processed data
        """
        data = self.read_raw_data(data_dir, add_instruction, task_instruction_idx)
        
        if "train" in data:
            train_data = data["train"]
            
            # Apply class balancing if requested
            if class_balanced and "label" in train_data.columns:
                train_data = self.balance_classes(train_data, "label")
            
            # Apply sampling limits
            if max_train_samples > 0 or max_train_percent > 0:
                train_data = self.sample_data(train_data, max_samples=max_train_samples,
                                           max_percent=max_train_percent)
            
            # Shuffle data if requested
            if shuffle_data:
                train_data = shuffle(train_data, random_state=self.random_state)
                
            data["train"] = train_data
            
        return data
        
    def balance_classes(self, data: pd.DataFrame, label_column: str,
                       method: str = 'undersample') -> pd.DataFrame:
        """
        Balance classes in the dataset.
        
        Args:
            data: Input DataFrame
            label_column: Name of the label column
            method: Balancing method ('undersample' or 'oversample')
            
        Returns:
            Balanced DataFrame
        """
        class_counts = data[label_column].value_counts()
        
        if method == 'undersample':
            # Undersample majority classes
            min_count = class_counts.min()
            balanced_data = []
            for label in class_counts.index:
                balanced_data.append(
                    data[data[label_column] == label].sample(
                        min_count, random_state=self.random_state
                    )
                )
            return pd.concat(balanced_data)
        
        elif method == 'oversample':
            # Oversample minority classes
            max_count = class_counts.max()
            balanced_data = []
            for label in class_counts.index:
                class_data = data[data[label_column] == label]
                if len(class_data) < max_count:
                    # Oversample with replacement
                    balanced_data.append(
                        class_data.sample(
                            max_count, replace=True, random_state=self.random_state
                        )
                    )
                else:
                    balanced_data.append(class_data)
            return pd.concat(balanced_data)
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")
            
    def sample_data(self, data: pd.DataFrame, max_samples: int = -1,
                   max_percent: float = -1, shuffle: bool = True) -> pd.DataFrame:
        """
        Sample data with size or percentage limits.
        
        Args:
            data: Input DataFrame
            max_samples: Maximum number of samples (-1 for no limit)
            max_percent: Maximum percentage of data to use (-1 for no limit)
            shuffle: Whether to shuffle before sampling
            
        Returns:
            Sampled DataFrame
        """
        if shuffle:
            data = shuffle(data, random_state=self.random_state)
            
        if max_samples > 0:
            return data.head(min(len(data), max_samples))
        elif max_percent > 0:
            n_samples = int(len(data) * max_percent)
            return data.head(n_samples)
        return data
        
    def serialize_sequence(self, sequence: Union[List, np.ndarray, pd.Series],
                         add_index: bool = False) -> str:
        """
        Serialize a numerical sequence into a string representation.
        
        Args:
            sequence: Input sequence (list, numpy array, or pandas Series)
            add_index: Whether to include index in serialization
            
        Returns:
            Serialized string representation
        """
        if isinstance(sequence, (pd.Series, pd.DataFrame)):
            values = sequence.values
        else:
            values = np.array(sequence)
            
        # Format each value
        formatted = []
        for i, val in enumerate(values):
            if np.isnan(val):
                val_str = self.nan_tok
            else:
                val_str = f"{val:.6g}"  # Use general format with 6 significant digits
                
            if add_index:
                formatted.append(f"{i}: {val_str}")
            else:
                formatted.append(val_str)
                
        # Join with separator
        sep = f" {self.sep_tok} " if self.sep_tok != "." else self.sep_tok
        return sep.join(formatted)
        
    def read_raw_data(self, data_dir: str, add_instruction: bool = False,
                      task_instruction_idx: int = 0) -> Dict:
        """
        Read raw data from directory and process according to task type.
        
        Args:
            data_dir: Directory containing the data
            add_instruction: Whether to add task-specific instructions
            task_instruction_idx: Index of task instruction to use
            
        Returns:
            Dictionary containing raw data
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise ValueError(f"Data directory {data_dir} does not exist")
            
        # Determine task type from directory structure
        task_type = self._determine_task_type(data_dir)
        
        if task_type == "entity_matching":
            return self._read_entity_matching_data(data_dir, add_instruction, task_instruction_idx)
        elif task_type == "imputation":
            return self._read_imputation_data(data_dir, add_instruction, task_instruction_idx)
        elif task_type == "error_detection":
            return self._read_error_detection_data(data_dir, add_instruction, task_instruction_idx)
        elif task_type == "schema_matching":
            return self._read_schema_matching_data(data_dir, add_instruction, task_instruction_idx)
        else:
            raise ValueError(f"Unknown task type for directory {data_dir}")
            
    def _determine_task_type(self, data_dir: Path) -> str:
        """Determine the task type from directory structure and contents."""
        # Implementation would depend on your specific directory structure
        # This is a placeholder that should be customized
        return "entity_matching"  # Default to entity matching
        
    def _read_entity_matching_data(self, data_dir: Path, add_instruction: bool,
                                 task_instruction_idx: int) -> Dict:
        """Read and process entity matching data."""
        splits = {}
        for split in ["train", "valid", "test"]:
            split_path = data_dir / f"{split}.csv"
            if not split_path.exists():
                continue
                
            # Read tables A and B
            tableA = pd.read_csv(data_dir / "tableA.csv")
            tableB = pd.read_csv(data_dir / "tableB.csv")
            
            # Process the data
            splits[split] = self.read_blocked_pairs(
                str(split_path), tableA, tableB,
                add_instruction=add_instruction,
                instruction=self._get_task_instruction("entity_matching", task_instruction_idx)
            )
            
        return splits
        
    def read_blocked_pairs(self, split_path: str, tableA: pd.DataFrame,
                          tableB: pd.DataFrame, cols_to_drop: List[str] = None,
                          col_renaming: Dict[str, str] = None, add_instruction: bool = False,
                          instruction: str = "", suffix: str = "", 
                          prod_name: str = "Product") -> pd.DataFrame:
        """
        Read and process blocked pairs for entity matching.
        
        Args:
            split_path: Path to the split file
            tableA: First table for matching
            tableB: Second table for matching
            cols_to_drop: Columns to drop from tables
            col_renaming: Dictionary for column renaming
            add_instruction: Whether to add instructions
            instruction: Instruction text
            suffix: Suffix to add to text
            prod_name: Name of the product/entity
            
        Returns:
            DataFrame containing processed pairs
        """
        if cols_to_drop is None:
            cols_to_drop = []
        if col_renaming is None:
            col_renaming = {}
            
        # Drop specified columns
        for c in cols_to_drop:
            if c in tableA.columns:
                tableA = tableA.drop(c, axis=1)
            if c in tableB.columns:
                tableB = tableB.drop(c, axis=1)
                
        # Rename columns if specified
        if col_renaming:
            tableA = tableA.rename(columns=col_renaming)
            tableB = tableB.rename(columns=col_renaming)
            
        # Create column maps
        column_mapA = {f"{c}_A": c for c in tableA.columns if c != "id"}
        column_mapB = {f"{c}_B": c for c in tableB.columns if c != "id"}
        
        # Read and process pairs
        labels = pd.read_csv(split_path)
        mergedA = pd.merge(labels, tableA, right_on="id", left_on="ltable_id")
        merged = pd.merge(
            mergedA, tableB,
            right_on="id", left_on="rtable_id",
            suffixes=("_A", "_B")
        )
        
        # Create text representation
        merged["text"] = merged.apply(
            lambda row: self._serialize_match_pair(
                row, column_mapA, column_mapB,
                add_instruction, instruction, suffix,
                prod_name
            ),
            axis=1
        )
        
        # Create label string
        merged["label_str"] = merged["label"].map({1: "Yes\n", 0: "No\n"})
        
        return merged
        
    def _serialize_match_pair(self, row: pd.Series, column_mapA: Dict[str, str],
                            column_mapB: Dict[str, str], add_instruction: bool,
                            instruction: str, suffix: str, prod_name: str) -> str:
        """Serialize a pair of entities for matching."""
        textA = self._serialize_row(row, column_mapA)
        textB = self._serialize_row(row, column_mapB)
        
        text = f"{prod_name} A is {textA}. {prod_name} B is {textB}.{suffix} "
        if add_instruction:
            text = f"{instruction} {text}"
            
        return text
        
    def _serialize_row(self, row: pd.Series, column_map: Dict[str, str]) -> str:
        """Serialize a single row of data."""
        parts = []
        for c_og, c_map in column_map.items():
            value = str(row[c_og]).strip()
            if value == "nan":
                value = self.nan_tok
            parts.append(f"{c_map}: {value}")
            
        sep = f" {self.sep_tok} " if self.sep_tok != "." else self.sep_tok
        return sep.join(parts)
        
    def _get_task_instruction(self, task_type: str, idx: int = 0) -> str:
        """Get instruction text for a specific task type."""
        # Implementation would depend on your instruction templates
        # This is a placeholder that should be customized
        return ""
