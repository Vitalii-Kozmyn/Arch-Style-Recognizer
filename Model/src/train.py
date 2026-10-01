import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

graphs_save_path = "/content/drive/MyDrive/arch_model_resnet18_vf_graphs.png"
matrix_save_path = "/content/drive/MyDrive/arch_model_resnet18_vf_matrix.png"
final_model_path = "/content/drive/MyDrive/arch_model_resnet18_vf_last.pth"
data_path        = "./dataset/architectural-styles-dataset"

# ──────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────
NUM_EPOCHS      = 30       
BATCH_SIZE      = 64
MIXUP_ALPHA     = 0.3      
CUTMIX_ALPHA    = 0.5      
MIXUP_PROB      = 0.5      
GRAD_CLIP       = 1.0      
LABEL_SMOOTHING = 0.1      
WARMUP_EPOCHS   = 3        

# ──────────────────────────────────────────────
# TRANSFORMS 
# ──────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),                                
    transforms.RandomRotation(15),                                       
    transforms.ColorJitter(brightness=0.3, contrast=0.3,          
                           saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.05),                            
    transforms.TrivialAugmentWide(),                               
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),          
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ──────────────────────────────────────────────
# DATASET SPLIT: 80% train / 20% test 
# ──────────────────────────────────────────────
full_train_dataset = datasets.ImageFolder(root=data_path, transform=train_transform)
full_test_dataset  = datasets.ImageFolder(root=data_path, transform=test_transform)
class_names        = full_train_dataset.classes
num_classes        = len(class_names)
num_data           = len(full_train_dataset)

train_size = int(0.80 * num_data)
test_size  = num_data - train_size

split_generator = torch.Generator().manual_seed(67)
train_dummy, test_dummy = random_split(
    range(num_data), [train_size, test_size], generator=split_generator
)

train_dataset = Subset(full_train_dataset, train_dummy.indices)
test_dataset  = Subset(full_test_dataset,  test_dummy.indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

# ──────────────────────────────────────────────
# MIXUP / CUTMIX HELPERS
# ──────────────────────────────────────────────
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def cutmix_data(x, y, alpha=CUTMIX_ALPHA):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    W, H = x.size(3), x.size(2)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = max(0, cx - cut_w // 2); x2 = min(W, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2); y2 = min(H, cy + cut_h // 2)
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed_x, y, y[index], lam

def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = True

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_ftrs, 256),
    nn.BatchNorm1d(256),
    nn.SiLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)   
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ──────────────────────────────────────────────
# CLASS WEIGHTS & LOSS
# ──────────────────────────────────────────────
print("⚖️ Обчислення ваг класів...")
train_targets = [full_train_dataset.targets[i] for i in train_dummy.indices]
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_targets),
    y=train_targets
)
class_weights = torch.tensor(class_weights_arr, dtype=torch.float).to(device)
loss_f = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

# ──────────────────────────────────────────────
# OPTIMIZER & SCHEDULER
# ──────────────────────────────────────────────
optimizer = optim.AdamW([
    {'params': model.layer1.parameters(), 'lr': 1e-6},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 5e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3, 'weight_decay': 1e-2},
], weight_decay=1e-2)

def warmup_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    return 1.0

warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
)

def step_scheduler(epoch):
    if epoch < WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()

# ──────────────────────────────────────────────
# INITIALIZE HISTORY
# ──────────────────────────────────────────────
best_test_acc = 0.0
history = {
    'train_loss': [], 'test_loss': [],
    'train_acc':  [], 'test_acc':  []
}
# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────
print("Початок навчання...\n")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss, train_correct = 0.0, 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        use_mix = np.random.rand() < MIXUP_PROB
        if use_mix:
            if np.random.rand() < 0.5:
                inputs, y_a, y_b, lam = mixup_data(inputs, labels)
            else:
                inputs, y_a, y_b, lam = cutmix_data(inputs, labels)
            outputs = model(inputs)
            loss = mixed_criterion(loss_f, outputs, y_a, y_b, lam)
        else:
            outputs = model(inputs)
            loss = loss_f(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        
        if use_mix:
            train_correct += (lam * (preds == y_a).float().sum() + (1 - lam) * (preds == y_b).float().sum())
        else:
            train_correct += (preds == labels).sum()

    step_scheduler(epoch)

    model.eval()
    test_loss, test_correct = 0.0, 0
    all_preds, all_labels   = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_f(outputs, labels)
            
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_train_loss = train_loss / len(train_loader)
    avg_test_loss  = test_loss  / len(test_loader)

    epoch_train_acc = torch.tensor(train_correct / len(train_dataset))
    epoch_test_acc  = test_correct.double()  / len(test_dataset)

    history['train_loss'].append(avg_train_loss)
    history['test_loss'].append(avg_test_loss)
    history['train_acc'].append(epoch_train_acc.item())
    history['test_acc'].append(epoch_test_acc.item())

    current_lr = optimizer.param_groups[-1]['lr']
    
    if epoch_test_acc > best_test_acc:
        best_test_acc = epoch_test_acc
    
    print(
        f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
        f"Train Acc: {epoch_train_acc:.4f} | Test Acc: {epoch_test_acc:.4f} | "
        f"Loss(test): {avg_test_loss:.4f} | LR: {current_lr:.7f}"
    )

# ──────────────────────────────────────────────
# FINAL METRICS 
# ──────────────────────────────────────────────
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)

final_test_loss = history['test_loss'][-1]
final_test_acc  = history['test_acc'][-1]
final_train_acc = history['train_acc'][-1]

print(f"\n{'='*55}")
print(f"  ФІНАЛЬНІ РЕЗУЛЬТАТИ V7 (ResNet-18)")
print(f"{'='*55}")
print(f"  Train Acc:        {final_train_acc:.4f}")
print(f"  Test Loss:        {final_test_loss:.4f}")
print(f"  Test Acc:         {final_test_acc:.4f}")
print(f"  Best Test Acc:    {best_test_acc:.4f}")
print(f"  Precision:        {precision:.4f}")
print(f"  Recall:           {recall:.4f}")
print(f"  F1 (macro):       {f1:.4f}")
print(f"{'='*55}")

# ──────────────────────────────────────────────
# PLOTS 
# ──────────────────────────────────────────────
actual_epochs = len(history['train_loss'])
epochs_range  = range(1, actual_epochs + 1)

plt.figure(figsize=(16, 5))

# Loss curves
plt.subplot(1, 3, 1)
plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='royalblue', linewidth=2)
plt.plot(epochs_range, history['test_loss'],  label='Test Loss',  color='crimson',   linewidth=2)
plt.title('Функція втрат')
plt.xlabel('Епоха'); plt.ylabel('Loss')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)

# Accuracy curves
plt.subplot(1, 3, 2)
plt.plot(epochs_range, history['train_acc'], label='Train Acc', color='royalblue', linewidth=2)
plt.plot(epochs_range, history['test_acc'],  label='Test Acc',  color='crimson',   linewidth=2)
plt.title('Точність')
plt.xlabel('Епоха'); plt.ylabel('Accuracy')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)

# Scatter: predictions
vis_labels = all_labels[:150]
vis_preds  = all_preds[:150]
plt.subplot(1, 3, 3)
plt.scatter(range(len(vis_labels)), vis_labels, label='Очікуваний', c='forestgreen', alpha=0.5, s=25)
plt.scatter(range(len(vis_preds)),  vis_preds,  label='Прогноз',    c='darkorange', marker='x', alpha=0.8, s=25)
plt.title('Перші 150 тестів')
plt.xlabel('Індекс'); plt.ylabel('Клас')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(graphs_save_path, dpi=150)
print(f"📈 Графіки збережено: {graphs_save_path}")
plt.show()

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(16, 14))
short_class_names = [c[:14] for c in class_names]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_class_names)
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, values_format='d')
plt.title(f'Матриця помилок (Test Acc: {final_test_acc:.4f})')
plt.tight_layout()
plt.savefig(matrix_save_path, dpi=150)
print(f"📊 Матриця помилок збережена: {matrix_save_path}")
plt.show()

torch.save(model.state_dict(), final_model_path)
print(f"✅ Фінальні ваги збережено: {final_model_path}")