import os
import logging
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, TrainingArguments, Trainer
from transformers import TrainerCallback
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Kiểm tra nếu GPU khả dụng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Đọc dữ liệu ===
train_file = r'C:\GPT2\Processed_UNSW_NB15_training_set.npz'
test_file = r'C:\GPT2\Processed_UNSW_NB15_testing_set.npz'

# Load dữ liệu từ file
train_data = np.load(train_file, allow_pickle=True)
test_data = np.load(test_file, allow_pickle=True)

X_train, y_train = train_data["X"], train_data["y"]
X_test, y_test = test_data["X"], test_data["y"]

# === Dùng 1 phần dữ liệu ngẫu nhiên tùy thực nghiệm ===
X_train_change, _, y_train_change, _ = train_test_split(X_train, y_train, test_size=0.5, stratify=y_train, random_state=None)
X_test_change, _, y_test_change, _ = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=None)

# === Chuyển dữ liệu thành chuỗi văn bản ===
def convert_to_text_batch(features, labels):
    """Chuyển đặc trưng mạng và nhãn thành chuỗi văn bản (theo batch)"""
    texts = []
    for x, y in zip(features, labels):
        text = (
            f"Source IP: {x[0]}, Destination IP: {x[1]}, Protocol: {x[2]}, "
            f"Service: {x[3]}, Source Bytes: {x[4]}, Destination Bytes: {x[5]}. "
            f"Label: {'Attack' if y == 1 else 'Normal'}."
        )
        texts.append(text)
    return texts

train_texts_change = convert_to_text_batch(X_train_change, y_train_change)
test_texts_change = convert_to_text_batch(X_test_change, y_test_change)

# === Load GPT-2 Tokenizer và Model ===
model_dir = "C:\GPT2\ModelGPT2"  # Đường dẫn đến thư mục mô hình
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token  # Đặt pad_token bằng eos_token

model = GPT2ForSequenceClassification.from_pretrained(model_dir, num_labels=6).to(device)
model.config.pad_token_id = tokenizer.pad_token_id

# === Mã hóa dữ liệu ===
def tokenize_batch(texts, labels):
    """Mã hóa dữ liệu theo batch"""
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",  # Giảm padding
        max_length=32,  # Giảm độ dài tối đa của chuỗi
        return_tensors="pt",
    )
    tokenized["labels"] = torch.tensor(labels, dtype=torch.long)
    return tokenized

# Tokenize dữ liệu
train_encoded_change = tokenize_batch(train_texts_change, y_train_change.tolist())
test_encoded_change = tokenize_batch(test_texts_change, y_test_change.tolist())

# Đường dẫn file log
log_dir = r"C:\GPT2\Checkpoint\(24-1-2025)_1"
log_file = os.path.join(log_dir, "training_log.csv")
os.makedirs(log_dir, exist_ok=True)

# Cấu hình log
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# === Định nghĩa callback để ghi lại thông số trong quá trình huấn luyện ===
class LogMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_message = f"{state.global_step},{logs.get('loss', 'N/A')},{logs.get('eval_accuracy', 'N/A')},{logs.get('eval_loss', 'N/A')}"
            logger.info(log_message)
            
    def on_train_begin(self, args, state, control, **kwargs):
        # Ghi header vào file log
        logging.info("step,loss,eval_accuracy,eval_loss")

# === Huấn luyện mô hình ===
training_args = TrainingArguments(
    output_dir=r'C:\GPT2\Checkpoint\(24-1-2025)_1',
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    max_grad_norm=0.5,
    logging_dir=os.path.join(log_dir, "logs"),
    logging_steps=500,
    load_best_model_at_end=True,
)

# Tạo Dataset từ Tensor
train_dataset_change = Dataset.from_dict(
    {"input_ids": train_encoded_change["input_ids"], 
     "attention_mask": train_encoded_change["attention_mask"], 
     "labels": train_encoded_change["labels"]}
)
test_dataset_change = Dataset.from_dict(
    {"input_ids": test_encoded_change["input_ids"], 
     "attention_mask": test_encoded_change["attention_mask"], 
     "labels": test_encoded_change["labels"]}
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_change,
    eval_dataset=test_dataset_change,
    callbacks=[LogMetricsCallback()]
)

trainer.train()

# Lưu mô hình đã tinh chỉnh
model_dir = r'C:\GPT2\Model\GPT-2_Model_Refinement_50%_2'

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print("Model fine-tuned and saved!")

# === Đánh giá mô hình ===
# Hàm dự đoán
def predict_batch(texts, model, tokenizer):
    """Dự đoán theo batch"""
    model.eval()
    results = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=32  # Giảm độ dài chuỗi để tiết kiệm bộ nhớ
            ).to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).cpu().item()
            results.append(predicted_class)
    return results

# Chạy dự đoán trên tập test
predicted_labels_change = predict_batch(test_texts_change, model, tokenizer)

# Đánh giá hiệu suất
print(classification_report(y_test_change, predicted_labels_change, target_names=["Attack_Type0", "Attack_Type5", "Attack_Type1", "Attack_Type2", "Attack_Type3", "Attack_Type4"]))

# === Chuẩn bị dữ liệu từ classification_report ===
report = classification_report(
    y_test_change,
    predicted_labels_change,
    target_names=["Attack_Type0", "Attack_Type5", "Attack_Type1", "Attack_Type2", "Attack_Type3", "Attack_Type4"],
    output_dict=True,
)

# Lấy dữ liệu cần thiết từ classification_report
precision_per_class = [report[class_name]["precision"] for class_name in report if class_name not in ["accuracy", "macro avg", "weighted avg"]]
recall_per_class = [report[class_name]["recall"] for class_name in report if class_name not in ["accuracy", "macro avg", "weighted avg"]]
f1_score_per_class = [report[class_name]["f1-score"] for class_name in report if class_name not in ["accuracy", "macro avg", "weighted avg"]]
accuracy_per_class = [report[class_name]["recall"] for class_name in report if class_name not in ["accuracy", "macro avg", "weighted avg"]]  # Accuracy tương đương recall
class_names = [class_name for class_name in report if class_name not in ["accuracy", "macro avg", "weighted avg"]]

# === Vẽ biểu đồ ===
plt.figure(figsize=(14, 10))

# Biểu đồ 1: Accuracy của từng Attack Type
plt.subplot(2, 1, 1)
plt.bar(class_names, accuracy_per_class, color='steelblue', alpha=0.8)
plt.title('Accuracy by Attack Type')
plt.ylabel('Accuracy')
plt.xlabel('Attack Type')
plt.ylim(0, 1.1)  # Giá trị nằm trong khoảng 0-1
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Biểu đồ 2: Precision, Recall, và F1-Score
bar_width = 0.25
indices = np.arange(len(class_names))

plt.subplot(2, 1, 2)
plt.bar(indices, precision_per_class, width=bar_width, label='Precision', color='skyblue')
plt.bar(indices + bar_width, recall_per_class, width=bar_width, label='Recall', color='lightgreen')
plt.bar(indices + 2 * bar_width, f1_score_per_class, width=bar_width, label='F1-Score', color='salmon')

# Cài đặt trục và chú thích
plt.xlabel('Attack Type')
plt.ylabel('Scores')
plt.title('Precision, Recall, and F1-Score by Attack Type')
plt.xticks(indices + bar_width, class_names, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Hiển thị cả hai biểu đồ
plt.tight_layout()
plt.show()

# === Đọc file log và vẽ đồ thị ===
log_data = pd.read_csv(log_file)

# Xử lý để lấy giá trị loss  
log_data['loss'] = pd.to_numeric(log_data['loss'], errors='coerce')
# Vẽ đồ thị
plt.figure(figsize=(12, 8))
plt.plot(log_data.index, log_data['loss'], label='Loss', color='blue', linewidth=2)
plt.title('Training Loss', fontsize=16)
plt.xlabel('Step', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

