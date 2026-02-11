import torch
import numpy as np
import shutil
from pathlib import Path
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

class BaseTransformerClassifier:
    def __init__(
        self,
        model,
        tokenizer,
        num_train_epochs=20,
        batch_size=16,
        use_cuda=True,
        problem_type="single_label_classification",  # or "multi_label_classification"
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )

        self.problem_type = problem_type
        self.tokenizer = tokenizer

        self.model = model.to(self.device)

        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.trainer = None
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size

    def _tokenize(self, batch):
        return self.tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=128,
        )

    def _prepare_dataset(self, df):
        dataset = Dataset.from_pandas(df.copy())
        dataset = dataset.map(self._tokenize, batched=True)

        dataset = dataset.remove_columns(["text"])
        dataset = dataset.rename_column("label", "labels")

        # Convert to torch tensors
        dataset.set_format("torch")

        if self.problem_type == "multi_label_classification":
            # Ensure labels are float32 for BCEWithLogitsLoss
            dataset = dataset.map(
                lambda x: {"labels": torch.tensor(x["labels"], dtype=torch.float32)}
            )

        return dataset
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        if self.problem_type == "multi_label_classification":
            # Apply sigmoid and threshold for multi-label
            predictions = (torch.sigmoid(torch.tensor(predictions)).numpy() >= 0.5).astype(int)
            f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        else:
            # Single-label classification
            predictions = np.argmax(predictions, axis=1)
            f1 = f1_score(labels, predictions, average='binary', zero_division=0)
        
        return {"f1": f1}
    
    def get_trainers(self, train_df):
        # Split into 80% train, 20% validation
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(train_df) * 0.8)
        train_split = train_df.iloc[:split_idx]
        val_split = train_df.iloc[split_idx:]
        
        train_dataset = self._prepare_dataset(train_split)
        val_dataset = self._prepare_dataset(val_split)
        
        training_args = TrainingArguments(
            output_dir="./checkpoints",
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            save_strategy="epoch",
            logging_strategy="epoch",
            eval_strategy="epoch",
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        warmup_args = TrainingArguments(
            output_dir="./results_warmup",
            num_train_epochs=1,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            save_strategy="no",
            logging_strategy="epoch",
            eval_strategy="epoch",
            report_to="none",
        )
        warmup_trainer = Trainer(
            model=self.model,
            args=warmup_args,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        return warmup_trainer, trainer

    def train_model(self, train_df):
        warmup_trainer, self.trainer = self.get_trainers(train_df)
        
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        warmup_trainer.train()
        
        for param in self.model.classifier.parameters():
            param.requires_grad = False
        for param in trainable_params:
            param.requires_grad = True
        
        self.trainer.train()

    def predict(self, texts, threshold=0.5):
        dataset = Dataset.from_dict({"text": texts})
        dataset = dataset.map(self._tokenize, batched=True)
        dataset.set_format("torch")

        outputs = self.trainer.predict(dataset)
        logits = outputs.predictions

        if self.problem_type == "multi_label_classification":
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            preds = (probs >= threshold).astype(int)
            return preds, probs
        else:
            preds = np.argmax(logits, axis=1)
            return preds, logits


class RobertaClassifier(BaseTransformerClassifier):
    def __init__(self, **kwargs):
        model_name = "roberta-base"
        problem_type = "single_label_classification"
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type=problem_type,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            problem_type=problem_type,
            **kwargs
        )


class RobertaMultiLabelClassifier(BaseTransformerClassifier):
    def __init__(self, **kwargs):
        model_name = "roberta-base"
        problem_type = "multi_label_classification"
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=7,
            problem_type=problem_type,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            problem_type=problem_type,
            **kwargs
        )
