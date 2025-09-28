from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import logging
import random

if TYPE_CHECKING:
    from config import RuntimeConfig, MockTokenizer

from config import load_model, load_tokenizer
from interfaces import Collator, Evaluator, EvaluationResult, Classifier, Trainer

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class MyEvaluationResult(EvaluationResult):
    accuracy: float
    total_samples: int
    correct_predictions: int
    
    def visualize(self) -> None:
        logger.info("visualizing_evaluation_results", extra={
            "accuracy": self.accuracy,
            "total_samples": self.total_samples,
            "correct_predictions": self.correct_predictions
        })
        print(f"Evaluation Results:\n  Accuracy: {self.accuracy:.3f}\n  Total Samples: {self.total_samples}\n  Correct Predictions: {self.correct_predictions}")
    
    def save(self, path: str) -> None:
        logger.info("saving_evaluation_results", extra={"path": path})
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Accuracy: {self.accuracy:.3f}\n")
            f.write(f"Total Samples: {self.total_samples}\n")
            f.write(f"Correct Predictions: {self.correct_predictions}\n")
        logger.info("evaluation_results_saved", extra={"path": path})

@dataclass(frozen=True) 
class MyEvaluator(Evaluator):
    labels: list[str]
    
    def evaluate(self, predictions: list[list[int]], targets: list[int]) -> EvaluationResult:
        logger.info("computing_classification_metrics", extra={
            "num_predictions": len(predictions),
            "num_targets": len(targets),
            "num_classes": len(self.labels)
        })
        
        correct_predictions = random.randint(0, len(predictions))
        accuracy = correct_predictions / len(predictions) if predictions else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "total_samples": len(predictions),
            "correct_predictions": correct_predictions
        }

        logger.info("metrics_computed", extra=metrics)
        return MyEvaluationResult(**metrics)

@dataclass(frozen=True)
class MyCollator(Collator):
    tokenizer: "MockTokenizer"
    max_length: int
    
    def collate(self, batch: list[dict]) -> dict[str, Any]:
        logger.debug("tokenizing_batch", extra={
            "batch_size": len(batch),
            "max_length": self.max_length
        })
        
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]
        
        tokenized = self.tokenizer.tokenize(
            texts, 
            max_length=self.max_length, 
            truncation=True, 
            padding=True
        )
        
        tokenized["labels"] = labels
        
        logger.debug("tokenization_complete", extra={
            "input_shape": (len(tokenized["input_ids"]), len(tokenized["input_ids"][0]) if tokenized["input_ids"] else 0),
            "num_labels": len(labels)
        })
        
        return tokenized

@dataclass(frozen=True)
class MyTrainer(Trainer):
    classifier: Classifier
    config: "RuntimeConfig"

    def train(
        self,
        train_data: list[dict], 
        val_data: list[dict], 
        collator: Collator, 
        evaluator: Evaluator, 
        epochs: int
    ) -> dict[str, Any]:
        logger.info("starting_ml_training", extra={
            "train_size": len(train_data),
            "val_size": len(val_data),
            "epochs": epochs,
            "device": self.config.device
        })
        
        best_accuracy = 0.0
        training_history = []
        
        for epoch in range(epochs):
            logger.info("epoch_start", extra={"epoch": epoch + 1, "total_epochs": epochs})
            train_loss = self._train_epoch(self.classifier, train_data, collator, epoch)
            val_metrics = self._validate_epoch(self.classifier, val_data, collator, evaluator, epoch)
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                logger.info("new_best_model", extra={"epoch": epoch + 1, "accuracy": best_accuracy})
            
            epoch_summary = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **val_metrics
            }
            training_history.append(epoch_summary)
            
            logger.info("epoch_complete", extra=epoch_summary)
        
        results = {
            "best_accuracy": best_accuracy,
            "final_accuracy": training_history[-1]["accuracy"] if training_history else 0.0,
            "training_history": training_history,
            "total_epochs": epochs
        }
        
        logger.info("ml_training_complete", extra={
            "best_accuracy": best_accuracy,
            "final_accuracy": results["final_accuracy"]
        })
        
        return results
    
    def _train_epoch(self, classifier: Classifier, train_data: list[dict], collator: Collator, epoch: int) -> float:
        batch_size = 8
        total_loss = 0.0
        num_batches = max(1, len(train_data) // batch_size)
        
        for _ in range(num_batches):
            batch_loss = random.uniform(0.5, 1.5) * (1.0 - epoch * 0.1)
            total_loss += batch_loss
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.debug("train_epoch_complete", extra={
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "batches": num_batches
        })
        
        return avg_loss

    def _validate_epoch(self, classifier: Classifier, val_data: list[dict], collator: Collator, evaluator: Evaluator, epoch: int) -> dict[str, float]:
        batch_size = 16
        all_predictions = []
        all_targets = []
        
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i + batch_size]
            collated = collator.collate(batch)
            
            predictions = classifier.classify(collated["input_ids"])
            all_predictions.extend(predictions)
            all_targets.extend(collated["labels"])

        evaluation_result = evaluator.evaluate(all_predictions, all_targets)
        
        metrics = {
            "accuracy": evaluation_result.accuracy,
            "total_samples": evaluation_result.total_samples,
            "correct_predictions": evaluation_result.correct_predictions
        }
        
        logger.debug("validation_epoch_complete", extra={
            "epoch": epoch + 1,
            **metrics
        })
        
        return metrics

class MyClassifier(Classifier):
    def __init__(self, model_name: str, labels: list[str], cfg: "RuntimeConfig"):
        self.model_name = model_name
        self.labels = labels
        self.num_classes = len(labels)
        self.cfg = cfg
        
        logger.debug("loading_classifier_resources", extra={
            "model_name": model_name,
            "device": cfg.device,
            "num_classes": self.num_classes
        })
        
        self.model = load_model(self.model_name, self.cfg.device, self.labels)
        self.tokenizer = load_tokenizer(self.model_name)
        
        logger.debug("classifier_resources_loaded")

    def classify(self, inputs: list[list[int]]) -> list[list[int]]:
        logger.debug("running_classification_inference", extra={"batch_size": len(inputs)})

        predictions = []
        for input_seq in inputs:
            pred = [random.randint(0, self.num_classes - 1) for _ in range(len(input_seq))]
            predictions.append(pred)

        return predictions
    
    def cleanup(self) -> None:
        logger.debug("cleaning_up_classifier_resources")
        if self.model:
            self.model.release()
        self.model = None
        self.tokenizer = None

    def run(self, data: Any) -> dict[str, Any]:
        logger.info("running_classification", extra={
            "model": self.model_name,
            "num_classes": self.num_classes,
            "data_type": type(data).__name__
        })

        if isinstance(data, dict) and "batch_size" in data:
            batch_size = data["batch_size"]
            predicted_class = batch_size % self.num_classes
        else:
            predicted_class = random.randint(0, self.num_classes - 1)
        
        result = {
            "predicted_class": predicted_class,
            "confidence": random.uniform(0.7, 0.95),
            "model": self.model_name,
            "num_classes": self.num_classes
        }
        
        logger.debug("classification_complete", extra=result)
        return result

def generate_simulation_data(data_size: int) -> tuple[list[dict], list[dict]]:
    logger.info("generating_simulation_data", extra={"data_size": data_size})
    
    templates = [
        ("This is positive text about {}", 0),
        ("Great example of {}", 0),
        ("Negative sentiment regarding {}", 1),
        ("Poor quality {}", 1),
        ("Neutral statement about {}", 2),
        ("Regular description of {}", 2),
    ]
    
    topics = ["machine learning", "python programming", "data science", "artificial intelligence", "software development"]
    
    def _create_data_batch(size: int) -> list[dict]:
        data = []
        for _ in range(size):
            template, label = random.choice(templates)
            topic = random.choice(topics)
            text = template.format(topic)
            data.append({"text": text, "label": label})
        return data
    
    train_size = int(data_size * 0.8)
    val_size = data_size - train_size
    
    train_data = _create_data_batch(train_size)
    val_data = _create_data_batch(val_size)
    
    logger.info("data_generation_complete", extra={
        "train_size": len(train_data),
        "val_size": len(val_data)
    })
    
    return train_data, val_data