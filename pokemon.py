import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
import os

# 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = './PokemonData'
num_epochs = 5
num_classes = len(os.listdir(data_dir))
results = {}

# 데이터 전처리 (실험 공통)
weights_std = models.ResNet18_Weights.DEFAULT
preprocess = weights_std.transforms()
full_dataset = datasets.ImageFolder(root=data_dir, transform=preprocess)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_db, test_db = torch.utils.data.random_split(full_dataset, [train_size, test_size])
train_loader = DataLoader(train_db, batch_size=32, shuffle=True)
test_loader = DataLoader(test_db, batch_size=32, shuffle=False)

def run_experiment(exp_name, model_type='resnet18', pretrained=True, freeze='all'):
    print(f"\n🚀 {exp_name} 시작...")
    
    # 모델 로드
    if model_type == 'resnet18':
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=w)
    else: # resnet34
        w = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=w)

    # 가중치 동결 설정
    if freeze == 'all':
        for param in model.parameters(): param.requires_grad = False
    elif freeze == 'partial':
        for param in model.parameters(): param.requires_grad = False
        for param in model.layer4.parameters(): param.requires_grad = True
    
    # 출력층 교체
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        history.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch+1} Loss: {history[-1]:.4f}")
    
    # 모델 저장
    torch.save(model.state_dict(), f"{exp_name}.pth")
    return history

# 5개 실험 실행
results['Exp1_R18_FC'] = run_experiment('Exp1', 'resnet18', True, 'all')
results['Exp2_R18_Full'] = run_experiment('Exp2', 'resnet18', True, 'none')
results['Exp3_R18_NoPre'] = run_experiment('Exp3', 'resnet18', False, 'none')
results['Exp4_R18_Partial'] = run_experiment('Exp4', 'resnet18', True, 'partial')
results['Exp5_R34_FC'] = run_experiment('Exp5', 'resnet34', True, 'all')

# Learning Curve 시각화
plt.figure(figsize=(10, 6))
for name, hist in results.items():
    plt.plot(range(1, num_epochs+1), hist, label=name)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve Comparison')
plt.legend()
plt.savefig('learning_curves.png')
plt.show()