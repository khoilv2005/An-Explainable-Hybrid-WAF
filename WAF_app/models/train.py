# -*- coding: utf-8 -*-
"""
🚀 WAF MODEL TRAINING - PHIÊN BẢN TỐI ƯU CHO >96% ACCURACY
============================================================
CÁC CẢI TIẾN:
1. ✅ Focal Loss để xử lý class imbalance
2. ✅ Label Smoothing để tránh overconfident predictions
3. ✅ Cosine Annealing với Warmup
4. ✅ Gradient Accumulation cho effective batch size lớn hơn
5. ✅ Mixed Precision Training (FP16)
6. ✅ SWA (Stochastic Weight Averaging) cho better generalization
7. ✅ Better Early Stopping dựa trên validation accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, SWALR
import pickle
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time
from tqdm import tqdm
import warnings
import math
warnings.filterwarnings('ignore')

# Import model
try:
    from model import WAF_Attention_Model, FocalLoss, LabelSmoothingBCE
except ImportError:
    print("❌ LỖI: Không tìm thấy file 'model.py'. Hãy kiểm tra lại.")
    exit()

# ==============================================================================
# CẤU HÌNH - ĐÃ TỐI ƯU CHO >96% ACCURACY
# ==============================================================================
DATA_FILE = "./data/processed_data.pkl"
MODEL_SAVE_PATH = "./data/waf_model.pth"
HISTORY_PATH = "./data/training_history.pkl"

# Hyperparameters được tối ưu
BATCH_SIZE = 64  # Giảm batch size để có more updates per epoch
EPOCHS = 30  # Cố định 30 epochs
LEARNING_RATE = 0.0005  # Giảm LR để học ổn định hơn
EMBEDDING_DIM = 128

# Gradient Accumulation - effective batch = BATCH_SIZE * ACCUMULATION_STEPS
ACCUMULATION_STEPS = 4  # Effective batch = 256

# Early Stopping - TẮT (chạy đủ 30 epochs)
USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.0005

# Warmup
WARMUP_EPOCHS = 5

# SWA (Stochastic Weight Averaging)
USE_SWA = True
SWA_START_EPOCH = 20  # Bắt đầu SWA sau epoch 20 (cho 30 epochs)
SWA_LR = 0.0001

# Loss Function
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.05

# Mixed Precision Training
USE_MIXED_PRECISION = True 

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def load_data():
    """Load dữ liệu đã tiền xử lý"""
    print(f"⏳ Loading data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print(f"❌ Lỗi: Không tìm thấy file {DATA_FILE}. Hãy chạy preprocess.py trước.")
        exit()

    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    X_train = torch.LongTensor(data['X_train'])
    X_test = torch.LongTensor(data['X_test'])
    y_train = torch.FloatTensor(data['y_train']).unsqueeze(1)
    y_test = torch.FloatTensor(data['y_test']).unsqueeze(1)
    vocab_size = data['vocab_size']

    print(f"✅ Data loaded!")
    print(f"   Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"   Vocab size: {vocab_size:,}")

    # Tính class weights cho Focal Loss
    attack_ratio = (y_train.sum()/len(y_train)).item()
    print(f"   Attack ratio (train): {attack_ratio:.2%}")

    return X_train, y_train, X_test, y_test, vocab_size, attack_ratio

def calculate_metrics(y_true, y_pred):
    """Tính toán các metrics"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return precision, recall, f1

def print_confusion_matrix(y_true, y_pred):
    """In confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    print("\n📊 Confusion Matrix:")
    print("                Predicted")
    print("              Normal  Attack")
    print(f"Actual Normal  {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"       Attack  {cm[1][0]:6d}  {cm[1][1]:6d}")

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f"\nFalse Positive Rate: {fpr:.2%}")
    print(f"False Negative Rate: {fnr:.2%}")

class CosineAnnealingWarmup:
    """Cosine Annealing với Linear Warmup"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

class EarlyStopping:
    """Early Stopping để tránh overfitting"""
    def __init__(self, patience=10, min_delta=0.0005, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        if self.mode == 'max':
            improved = (score - self.best_score) > self.min_delta
        else:
            improved = (self.best_score - score) > self.min_delta

        if improved:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

# ==============================================================================
# TRAINING FUNCTION - TỐI ƯU CHO >96%
# ==============================================================================
def train_epoch(model, train_loader, criterion, optimizer, scaler, device, accumulation_steps=1):
    """Training 1 epoch với Gradient Accumulation"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc="Training", leave=False)

    for i, (X_batch, y_batch) in enumerate(pbar):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Mixed Precision Training
        if USE_MIXED_PRECISION and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        # outputs là logits, cần sigmoid để có probability
        predicted = (torch.sigmoid(outputs) > 0.5).float().detach()
        all_preds.append(predicted.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    # Handle remaining gradients
    if len(train_loader) % accumulation_steps != 0:
        if USE_MIXED_PRECISION and device.type == 'cuda':
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    avg_loss = total_loss / len(train_loader)
    accuracy = (all_preds == all_labels).mean()
    precision, recall, f1 = calculate_metrics(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1

def evaluate(model, test_loader, criterion, device):
    """Đánh giá model trên test set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            # outputs là logits, cần sigmoid để có probability
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_probs.append(torch.sigmoid(outputs).cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_probs = np.concatenate(all_probs).flatten()

    avg_loss = total_loss / len(test_loader)
    accuracy = (all_preds == all_labels).mean()
    precision, recall, f1 = calculate_metrics(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels, all_probs

# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================
def main():
    print("="*70)
    print("🚀 WAF MODEL TRAINING - PHIÊN BẢN TỐI ƯU CHO >96% ACCURACY")
    print("="*70)
    print(f"\n🔧 Thiết bị: {DEVICE}")
    if USE_MIXED_PRECISION and DEVICE.type == 'cuda':
        print("⚡ Mixed Precision Training: ENABLED")
    print(f"📦 Effective Batch Size: {BATCH_SIZE * ACCUMULATION_STEPS}")

    # 1. Load Data
    X_train, y_train, X_test, y_test, vocab_size, attack_ratio = load_data()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == 'cuda'),
        drop_last=True  # Drop last incomplete batch cho gradient accumulation
    )

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,  # Larger batch for evaluation
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == 'cuda')
    )

    # 2. Khởi tạo Model
    print("\n🏗️  Khởi tạo model...")
    model = WAF_Attention_Model(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_classes=1,
        dropout=0.3
    ).to(DEVICE)

    # Đếm parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")

    # 3. Loss Function
    if USE_FOCAL_LOSS:
        print(f"📉 Loss: Focal Loss (α={FOCAL_ALPHA}, γ={FOCAL_GAMMA})")
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    else:
        print(f"📉 Loss: Label Smoothing BCE (smoothing={LABEL_SMOOTHING})")
        criterion = LabelSmoothingBCE(smoothing=LABEL_SMOOTHING)

    # 4. Optimizer với weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # 5. Learning Rate Scheduler với Warmup
    scheduler = CosineAnnealingWarmup(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=EPOCHS,
        min_lr=1e-6
    )

    # 6. SWA Model
    if USE_SWA:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
        print(f"🔄 SWA: Enabled (starts at epoch {SWA_START_EPOCH})")

    # 7. Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION and DEVICE.type == 'cuda' else None

    # 8. Early Stopping (optional)
    if USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=MIN_DELTA,
            mode='max'
        )
    else:
        early_stopping = None

    # 9. Training History
    history = {
        'train_loss': [], 'train_accuracy': [], 'train_precision': [],
        'train_recall': [], 'train_f1': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [],
        'val_recall': [], 'val_f1': [],
        'learning_rates': []
    }

    best_acc = 0.0
    best_f1 = 0.0
    best_epoch = 0
    start_time = time.time()

    # ==============================================================================
    # TRAINING LOOP
    # ==============================================================================
    print("\n" + "="*70)
    print("🚀 BẮT ĐẦU TRAINING")
    print("="*70)

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # Update LR
        if USE_SWA and epoch >= SWA_START_EPOCH:
            current_lr = SWA_LR
        else:
            current_lr = scheduler.step(epoch)

        # TRAIN
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE, ACCUMULATION_STEPS
        )

        # SWA Update
        if USE_SWA and epoch >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # EVALUATE
        val_loss, val_acc, val_prec, val_rec, val_f1, val_preds, val_labels, _ = evaluate(
            model, test_loader, criterion, DEVICE
        )

        # Save history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['train_precision'].append(train_prec)
        history['train_recall'].append(train_rec)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        history['learning_rates'].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Print metrics
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] | Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
        print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print(f"        | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

        # Save best model based on accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  💾 Saved best model! (Acc: {best_acc:.4f}, F1: {best_f1:.4f})")

        # Early Stopping (if enabled)
        if early_stopping is not None:
            if early_stopping(val_acc, model):
                print(f"\n⏹️  Early Stopping triggered at epoch {epoch+1}")
                print(f"   Best Accuracy: {best_acc:.4f} at epoch {best_epoch}")
                break

        # Check target
        if val_acc >= 0.96:
            print(f"\n🎯 TARGET REACHED! Accuracy: {val_acc:.4f} >= 96%")

    # ==============================================================================
    # FINAL EVALUATION WITH SWA
    # ==============================================================================
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("✅ TRAINING HOÀN TẤT")
    print("="*70)
    print(f"⏱️  Tổng thời gian: {total_time/60:.1f} phút")
    print(f"🏆 Best Accuracy: {best_acc:.4f} (Epoch {best_epoch})")
    print(f"🏆 Best F1 Score: {best_f1:.4f}")

    # Load best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # Evaluate with SWA model if available
    if USE_SWA:
        print("\n📊 ĐÁNH GIÁ VỚI SWA MODEL:")
        # Update batch normalization statistics
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)

        swa_loss, swa_acc, swa_prec, swa_rec, swa_f1, _, _, _ = evaluate(
            swa_model, test_loader, criterion, DEVICE
        )
        print(f"  SWA Accuracy:  {swa_acc:.4f}")
        print(f"  SWA Precision: {swa_prec:.4f}")
        print(f"  SWA Recall:    {swa_rec:.4f}")
        print(f"  SWA F1 Score:  {swa_f1:.4f}")

        # Save SWA model if better
        if swa_acc > best_acc:
            print(f"\n💾 SWA model is better! Saving...")
            torch.save(swa_model.module.state_dict(), MODEL_SAVE_PATH)
            best_acc = swa_acc
            best_f1 = swa_f1

    # Final evaluation
    print(f"\n📊 ĐÁNH GIÁ CUỐI CÙNG (Best Model):")
    final_loss, final_acc, final_prec, final_rec, final_f1, final_preds, final_labels, _ = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print(f"  Accuracy:  {final_acc:.4f}")
    print(f"  Precision: {final_prec:.4f}")
    print(f"  Recall:    {final_rec:.4f}")
    print(f"  F1 Score:  {final_f1:.4f}")

    # Confusion Matrix
    print_confusion_matrix(final_labels, final_preds)

    # Save training history
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history, f)
    print(f"\n💾 Đã lưu training history tại: {HISTORY_PATH}")

    # Summary
    print("\n" + "="*70)
    if best_acc >= 0.96:
        print("🎉 THÀNH CÔNG! ĐÃ ĐẠT MỤC TIÊU >96% ACCURACY!")
    else:
        print(f"📈 Accuracy hiện tại: {best_acc:.2%}")
        print("💡 Gợi ý: Thử tăng epochs hoặc điều chỉnh dữ liệu training")
    print("="*70)
    print("\n📋 BƯỚC TIẾP THEO:")
    print("="*70)
    print("1. python report.py     # Xem báo cáo chi tiết + biểu đồ")
    print("2. python explain.py    # Giải thích predictions")
    print("="*70)

if __name__ == "__main__":
    main()
