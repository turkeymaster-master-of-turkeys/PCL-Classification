import torch
import torch.nn.functional as F
import numpy as np
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


class FocalLossTrainer(Trainer):
    def __init__(self, *args, alpha=0.25, gamma=2.0, problem_type="single_label_classification", **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.problem_type = problem_type
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float()  # shape (B,8)
        outputs = model(**inputs)
        logits = outputs.logits                # shape (B, 8)

        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')  # (B,)
        probs = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probs, 1 - probs)                                  # (B,)

        alpha_t = self.alpha.view(1, -1)   # shape (B,)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        loss = focal_loss.mean()
        
        return (loss, outputs) if return_outputs else loss


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
            labels = labels[:, -1]
            predictions = predictions[:, -1]
            # Apply sigmoid and threshold for multi-label
            predictions = (torch.sigmoid(torch.tensor(predictions)).numpy() >= 0.5).astype(int)
            f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        else:
            # Single-label classification
            predictions = np.argmax(predictions, axis=1)
            f1 = f1_score(labels, predictions, average='binary', zero_division=0)
        
        return {"f1": f1}
    
    def _calculate_alpha(self, train_df):
        """Calculate alpha based on class distribution in training data."""
        if self.problem_type == "multi_label_classification":
            # For multi-label, calculate alpha for each label
            labels_array = np.array(train_df['label'].tolist())
            num_samples = len(labels_array)
            num_labels = labels_array.shape[1]
            
            alphas = []
            for i in range(num_labels):
                pos_count = labels_array[:, i].sum()
                neg_count = num_samples - pos_count
                # Calculate alpha as the ratio of negative to positive samples
                if pos_count > 0:
                    alpha = neg_count / pos_count
                else:
                    alpha = 1.0
                alphas.append(min(alpha, 10.0))  # Clip to 10
            
            return torch.tensor(alphas, dtype=torch.float32)
        else:
            # For single-label, calculate alpha for minority class
            label_counts = train_df['label'].value_counts()
            
            if len(label_counts) > 1:
                # Calculate inverse frequency, normalized
                max_count = label_counts.max()
                min_count = label_counts.min()
                alpha = min(max_count / min_count, 10.0)  # Clip to 10
            else:
                alpha = 1.0
            
            return alpha
    
    def get_trainers(self, train_df):
        # Split into 80% train, 20% validation
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(train_df) * 0.8)
        train_split = train_df.iloc[:split_idx]
        val_split = train_df.iloc[split_idx:]
        
        # Calculate alpha from training split
        alpha = self._calculate_alpha(train_split)
        if self.problem_type == "multi_label_classification":
            alpha = alpha.to(self.device)
        
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
        
        trainer = FocalLossTrainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            problem_type=self.problem_type,
            alpha=alpha,
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
        warmup_trainer = FocalLossTrainer(
            model=self.model,
            args=warmup_args,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            problem_type=self.problem_type,
            alpha=alpha,
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
            num_labels=8,
            problem_type=problem_type,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            problem_type=problem_type,
            **kwargs
        )


class AttentionJointClassifier(BaseTransformerClassifier):
    def __init__(self, **kwargs):
        model_name = "roberta-base"
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        latent_size = 32
        num_categories = 7
        
        class JointClassifier(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.latent_proj = torch.nn.ModuleList([torch.nn.Linear(base_model.config.hidden_size, latent_size) for _ in range(num_categories + 1)])
                self.attn = torch.nn.MultiheadAttention(embed_dim=latent_size, num_heads=1, batch_first=True)
                self.category_classifiers = torch.nn.ModuleList([torch.nn.Linear(latent_size, 1) for _ in range(num_categories)])
                self.pcl_classifier = torch.nn.Linear(latent_size * (num_categories + 1), 1)
                # Group classification layers for easier access during training
                self.classifier = torch.nn.ModuleDict({
                    'latent_proj': self.latent_proj,
                    'attn': self.attn,
                    'category': self.category_classifiers,
                    'pcl': self.pcl_classifier
                })

            def forward(self, input_ids, attention_mask, **kwargs):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state[:, 0]
                latents = [proj(pooled_output) for proj in self.latent_proj]
                category_latents = latents[:-1]
                pcl_latent = latents[-1]
                category_latents = torch.stack(category_latents, dim=1)  # (batch, 7, latent_size)
                category_residuals, _ = self.attn(category_latents, category_latents, category_latents)
                category_latents = category_latents + category_residuals  # (batch, 7, latent_size)
                # Apply separate classifier to each category latent
                category_logits = [classifier(category_latents[:, i, :]) for i, classifier in enumerate(self.category_classifiers)]
                category_logits = torch.cat(category_logits, dim=1)  # (batch, 7)
                # Flatten category latents for PCL classifier
                category_latents_flat = category_latents.view(category_latents.size(0), -1)
                pcl_logits = self.pcl_classifier(torch.cat([category_latents_flat, pcl_latent], dim=1))
                logits = torch.cat([category_logits, pcl_logits], dim=1)  # (batch, 8)
                
                # Return output in the same format as transformers models
                return type('obj', (object,), {'logits': logits})()
            
        super().__init__(
            model=JointClassifier(base_model),
            tokenizer=tokenizer,
            **kwargs
        )

