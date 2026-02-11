import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

class BaseTransformerClassifier:
    def __init__(
        self,
        model_name="roberta-base",
        num_labels=2,
        num_train_epochs=20,
        batch_size=16,
        use_cuda=True,
        problem_type="single_label_classification",  # or "multi_label_classification"
    ):
        self.problem_type = problem_type

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type=problem_type,
        ).to(self.device)

        self.training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_strategy="no",
            logging_strategy="epoch",
            report_to="none",
        )

        self.data_collator = DataCollatorWithPadding(self.tokenizer)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
        )

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


    def train_model(self, train_df):
        dataset = self._prepare_dataset(train_df)
        self.trainer.train_dataset = dataset
        self.trainer.train()
        self.model.save_pretrained("./checkpoints/best_model.pt")

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
        super().__init__(
            problem_type="single_label_classification",
            **kwargs
        )


class RobertaMultiLabelClassifier(BaseTransformerClassifier):
    def __init__(self, **kwargs):
        super().__init__(
            problem_type="multi_label_classification",
            **kwargs
        )
