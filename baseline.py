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
)


class FocalLossTrainer(Trainer):
    def __init__(self, *args, alpha=4.4, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float()  # shape (B,8)
        outputs = model(**inputs)
        logits = outputs['logits']                # shape (B, 8)

        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')  # (B,)
        probs = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probs, 1 - probs)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        task_weights = torch.ones(8, device=logits.device)
        task_weights[:-1] = 0.5 / 7
        task_weights[-1]  = 0.5 
        weighted = focal_loss * task_weights
        
        loss = weighted.sum(dim=1).mean()
        
        return (loss, outputs) if return_outputs else loss


class BaseTransformerClassifier:
    def __init__(
        self,
        model,
        tokenizer,
        num_train_epochs=20,
        batch_size=16,
        use_cuda=True
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )

        self.tokenizer = tokenizer

        self.model = model.to(self.device)

        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.trainer = None
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        
        self.val_dataset = None
        self.threshold = 0.5

    def _tokenize(self, batch):
        return self.tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=256,
        )

    def _prepare_dataset(self, df):
        dataset = Dataset.from_pandas(df.copy())
        dataset = dataset.map(self._tokenize, batched=True)

        dataset = dataset.remove_columns(["text"])
        dataset = dataset.rename_column("label", "labels")

        # Convert to torch tensors
        dataset.set_format("torch")

        # Ensure labels are float32 for BCEWithLogitsLoss
        dataset = dataset.map(
            lambda x: {"labels": torch.tensor(x["labels"], dtype=torch.float32)}
        )

        return dataset
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        labels = labels[:, -1]
        predictions = predictions[:, -1]
        # Apply sigmoid and threshold for multi-label
        predictions = (torch.sigmoid(torch.tensor(predictions)).numpy() >= 0.5).astype(int)
        f1 = f1_score(labels, predictions, average='binary', zero_division=0)
        
        return {"f1": f1}
    
    def _calculate_alpha(self, train_df):
        """Calculate alpha based on class distribution in training data."""
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
    
    def get_trainers(self, train_df, val_df):

        # Calculate alpha from training split
        alpha = self._calculate_alpha(train_df).to(self.device)
        
        train_dataset = self._prepare_dataset(train_df)
        val_dataset = self._prepare_dataset(val_df)
        
        training_args = TrainingArguments(
            output_dir="./checkpoints",
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            save_strategy="best",
            logging_strategy="epoch",
            eval_strategy="epoch",
            report_to="none",
            metric_for_best_model="f1",
            greater_is_better=True,
            learning_rate=1e-4,
        )
        
        trainer = FocalLossTrainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            alpha=alpha,
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
            learning_rate=1e-4,
        )
        warmup_trainer = FocalLossTrainer(
            model=self.model,
            args=warmup_args,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            alpha=alpha,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        self.val_dataset = val_dataset

        return warmup_trainer, trainer

    def train_model(self, train_df, val_df):
        warmup_trainer, self.trainer = self.get_trainers(train_df, val_df)
        
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
        
        self.calibrate_threshold()
        
    def calibrate_threshold(self):
        outputs = self.trainer.predict(self.val_dataset)
        logits = outputs.predictions[:, -1]
        labels = outputs.label_ids[:, -1]

        probs = torch.sigmoid(torch.tensor(logits)).numpy()

        best_threshold = 0.5
        best_f1 = 0.0

        # Sweep thresholds
        for t in np.linspace(0.01, 0.99, 99):
            preds = (probs >= t).astype(int)
            f1 = f1_score(labels, preds, average="binary", zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        print(f"Best threshold: {best_threshold:.3f}")
        print(f"Best validation micro-F1: {best_f1:.4f}")

        self.threshold = best_threshold

    def predict(self, texts):
        dataset = Dataset.from_dict({"text": texts})
        dataset = dataset.map(self._tokenize, batched=True)
        dataset.set_format("torch")

        outputs = self.trainer.predict(dataset)
        logits = outputs.predictions

        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs >= self.threshold).astype(int)
        return preds, probs


class RobertaMultiLabelClassifier(BaseTransformerClassifier):
    def __init__(self, **kwargs):
        model_name = "roberta-base"
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=8,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            **kwargs
        )


class AttentionJointClassifier(BaseTransformerClassifier):
    def __init__(self, classifier_type='attn', **kwargs):
        model = AttentionJointClassifier.get_model(classifier_type)
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            **kwargs
        )
        
    @classmethod
    def get_model(cls, classifier_type='attn'):
        latent_size = 32
        num_categories = 7
        model_name = "roberta-base"
        from transformers import AutoModel
        import peft
        base_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        
        peft_config = peft.LoraConfig(
            r=16,
            lora_alpha=32,
            task_type=peft.TaskType.TOKEN_CLS,
            target_modules=["query", "value", "key"]
        )
        base_model = peft.get_peft_model(base_model, peft_config)
        
        class AttnClassifier(torch.nn.Module):
            def __init__(self, latent_size, num_categories):
                super().__init__()
                self.latent_proj = torch.nn.ModuleList([torch.nn.Linear(768, latent_size) for _ in range(num_categories + 1)])
                self.attn = torch.nn.MultiheadAttention(embed_dim=latent_size, num_heads=1, batch_first=True)
                self.category_classifiers = torch.nn.ModuleList([torch.nn.Linear(latent_size, 1) for _ in range(num_categories)])
                self.pcl_classifier = torch.nn.Linear(latent_size * (num_categories + 1), 1)
                
            def forward(self, x):
                latents = [proj(x) for proj in self.latent_proj]
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
                return logits
        
        class ProjClassifier(torch.nn.Module):
            def __init__(self, latent_size, num_categories):
                super().__init__()
                self.latent_proj = torch.nn.Linear(768, latent_size)
                self.out_proj = torch.nn.Linear(latent_size, 8)
            
            def forward(self, x):
                x = self.latent_proj(x)
                return self.out_proj(x)
        
        class JointModel(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                if classifier_type == 'attn':
                    self.classifier = AttnClassifier(latent_size, num_categories)
                else:
                    self.classifier = ProjClassifier(latent_size, num_categories)
            
            def forward(self, input_ids, attention_mask, labels=None, **kwargs):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = self.classifier(outputs.last_hidden_state[:, 0])
                return {"logits": logits}

        return JointModel(base_model)


class EnsembleClassifier(BaseTransformerClassifier):
    def __init__(self, classifier_type='attn', num_models=3, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        
        class EnsembleModel(torch.nn.Module):
            def __init__(self, classifier_type, num_models):
                super().__init__()
                self.models = [AttentionJointClassifier.get_model(classifier_type) for _ in range(num_models)]
                self.model_modules = torch.nn.ModuleList([model.base_model for model in self.models])
                self.classifier = torch.nn.ModuleList([model.classifier for model in self.models])
            def forward(self, input_ids, attention_mask, labels=None, **kwargs):
                model_outputs = [model(input_ids, attention_mask, labels, **kwargs) for model in self.models]
                logits = [o['logits'] for o in model_outputs]
                logits = torch.stack(logits, dim=0).sum(dim=0)
                return {"logits": logits}
        
        super().__init__(
            model=EnsembleModel(classifier_type, num_models),
            tokenizer=tokenizer,
            **kwargs
        )
