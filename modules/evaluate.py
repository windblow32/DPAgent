"""Misc utils."""
import logging
from pathlib import Path
from typing import List

def setup_logger(log_dir: str):
    """Create log directory and logger."""
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    log_path = str(Path(log_dir) / "log.txt")
    
    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s [%(module)s] [%(levelname)s] %(message)s"
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def compute_metrics(preds: List, golds: List, task: str):
    """Compute metrics."""
    mets = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "crc": 0, "total": 0}
    for pred, label in zip(preds, golds):
        label = label.strip().lower()
        pred = pred.strip().lower()
        mets["total"] += 1
        if task in {
            "data_imputation",
            "entity_matching",
        }:
            crc = pred == label
        elif task in {"entity_matching", "schema_matching", "error_detection_spelling"}:
            crc = pred.startswith(label)
        elif task in {"error_detection"}:
            pred = pred.split("\n\n")[-1]
            crc = pred.endswith(label)
        else:
            raise ValueError(f"Unknown task: {task}")
        # Measure equal accuracy for generation
        if crc:
            mets["crc"] += 1
        if label == "yes":
            if crc:
                mets["tp"] += 1
            else:
                mets["fn"] += 1
        elif label == "no":
            if crc:
                mets["tn"] += 1
            else:
                mets["fp"] += 1

    prec = mets["tp"] / max(1, (mets["tp"] + mets["fp"]))
    rec = mets["tp"] / max(1, (mets["tp"] + mets["fn"]))
    acc = mets["crc"] / mets["total"]
    f1 = 2 * prec * rec / max(1, (prec + rec))
    return prec, rec, acc, f1


"""Data augmentation and evaluation utilities."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

class DataAugmentor:
    """Data augmentation and evaluation for data processing tasks."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize DataAugmentor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def augment_data(self, data: Union[List, np.ndarray],
                    methods: List[str] = None,
                    augment_ratio: float = 0.2) -> Union[List, np.ndarray]:
        """
        Augment data using specified methods.
        
        Args:
            data: Input data (list or numpy array)
            methods: List of augmentation methods to apply
            augment_ratio: Ratio of samples to augment
            
        Returns:
            Augmented data
        """
        if methods is None:
            methods = ['synonym_replacement', 'back_translation']
            
        n_samples = int(len(data) * augment_ratio)
        augmented_data = []
        
        for method in methods:
            if method == 'synonym_replacement':
                aug_samples = self._augment_with_synonyms(data, n_samples)
            elif method == 'back_translation':
                aug_samples = self._augment_with_translation(data, n_samples)
            else:
                logger.warning(f"Unknown augmentation method: {method}")
                continue
                
            if aug_samples:
                augmented_data.extend(aug_samples)
            
        if augmented_data:
            if isinstance(data, np.ndarray):
                return np.concatenate([data, np.array(augmented_data)])
            return data + augmented_data
        return data
        
    def _augment_with_synonyms(self, data: Union[List, np.ndarray],
                              n_samples: int) -> List:
        """Augment data using synonym replacement."""
        # TODO: Implement synonym replacement
        # This would use a thesaurus or word embedding model
        logger.info("Synonym replacement not implemented yet")
        return []
        
    def _augment_with_translation(self, data: Union[List, np.ndarray],
                                n_samples: int) -> List:
        """Augment data using back translation."""
        # TODO: Implement back translation
        # This would use translation APIs
        logger.info("Back translation not implemented yet")
        return []
        
    def compute_metrics(self, true_labels: Union[List, np.ndarray],
                       pred_labels: Union[List, np.ndarray],
                       task_type: str = 'classification') -> Dict:
        """
        Compute evaluation metrics.
        
        Args:
            true_labels: Ground truth labels
            pred_labels: Predicted labels
            task_type: Type of task ('classification' or 'matching')
            
        Returns:
            Dictionary of metrics
        """
        # Convert inputs to numpy arrays
        y_true = np.array(true_labels)
        y_pred = np.array(pred_labels)
        
        # Calculate basic metrics
        correct = (y_true == y_pred)
        accuracy = np.mean(correct)
        
        # For binary classification
        if task_type in ['classification', 'matching']:
            # Convert to boolean if needed
            if y_true.dtype != bool:
                y_true = y_true.astype(bool)
            if y_pred.dtype != bool:
                y_pred = y_pred.astype(bool)
                
            # Calculate TP, FP, TN, FN
            tp = np.sum((y_true == True) & (y_pred == True))
            fp = np.sum((y_true == False) & (y_pred == True))
            tn = np.sum((y_true == False) & (y_pred == False))
            fn = np.sum((y_true == True) & (y_pred == False))
            
            # Calculate metrics
            precision = tp / max(1, (tp + fp))
            recall = tp / max(1, (tp + fn))
            f1 = 2 * precision * recall / max(1, (precision + recall))
            
            # Create confusion matrix
            confusion = np.array([[tn, fp], [fn, tp]])
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'confusion_matrix': confusion.tolist(),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            }
        
        # For multi-class classification
        else:
            # Get unique classes
            classes = np.unique(np.concatenate([y_true, y_pred]))
            n_classes = len(classes)
            
            # Initialize confusion matrix
            confusion = np.zeros((n_classes, n_classes), dtype=int)
            for i, true_class in enumerate(classes):
                for j, pred_class in enumerate(classes):
                    confusion[i, j] = np.sum(
                        (y_true == true_class) & (y_pred == pred_class)
                    )
            
            # Calculate per-class metrics
            precisions = []
            recalls = []
            f1s = []
            
            for i in range(n_classes):
                tp = confusion[i, i]
                fp = np.sum(confusion[:, i]) - tp
                fn = np.sum(confusion[i, :]) - tp
                
                prec = tp / max(1, (tp + fp))
                rec = tp / max(1, (tp + fn))
                f1 = 2 * prec * rec / max(1, (prec + rec))
                
                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)
            
            return {
                'accuracy': float(accuracy),
                'precision': float(np.mean(precisions)),
                'recall': float(np.mean(recalls)),
                'f1': float(np.mean(f1s)),
                'confusion_matrix': confusion.tolist(),
                'per_class_precision': [float(p) for p in precisions],
                'per_class_recall': [float(r) for r in recalls],
                'per_class_f1': [float(f) for f in f1s]
            }
        
    def analyze_errors(self, data: Union[List, np.ndarray],
                      true_labels: Union[List, np.ndarray],
                      pred_labels: Union[List, np.ndarray]) -> Tuple[List, Dict]:
        """
        Analyze prediction errors.
        
        Args:
            data: Original data samples
            true_labels: True labels
            pred_labels: Predicted labels
            
        Returns:
            Tuple of (error_examples, error_stats)
        """
        # Convert to numpy arrays
        y_true = np.array(true_labels)
        y_pred = np.array(pred_labels)
        
        # Find errors
        error_mask = y_true != y_pred
        error_indices = np.where(error_mask)[0]
        
        # Get error examples
        error_examples = []
        for idx in error_indices:
            error_examples.append({
                'data': data[idx],
                'true_label': true_labels[idx],
                'pred_label': pred_labels[idx]
            })
        
        # Compute error statistics
        error_stats = {
            'total_errors': len(error_examples),
            'error_rate': len(error_examples) / len(data)
        }
        
        # Count error types
        error_types = {}
        for ex in error_examples:
            error_key = f"{ex['true_label']}->{ex['pred_label']}"
            error_types[error_key] = error_types.get(error_key, 0) + 1
            
        error_stats['error_types'] = error_types
        
        return error_examples, error_stats