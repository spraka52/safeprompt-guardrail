import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from training.dataset import get_label_columns

LABEL_COLUMNS = get_label_columns()


class SafePromptPredictor:
    def __init__(self, model_dir: str, max_length: int = 128):
        self.label_columns = LABEL_COLUMNS
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

    def predict(self, text: str) -> dict[str, float]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.sigmoid(logits).squeeze().cpu().tolist()

        # squeeze() returns a scalar when batch=1 and num_labels=1; ensure list
        if isinstance(probs, float):
            probs = [probs]

        return {label: round(prob, 4) for label, prob in zip(self.label_columns, probs)}
