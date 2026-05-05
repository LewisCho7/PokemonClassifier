import torch
from torchvision import models, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score
import os

# 설정 (학습 코드와 동일해야 함)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = './PokemonData'
weights_std = models.ResNet18_Weights.DEFAULT
preprocess = weights_std.transforms()
num_classes = len(os.listdir(data_dir))

# 테스트 데이터 로드 (학습 때와 동일한 분할을 위해 random_seed 고정 권장)
full_dataset = datasets.ImageFolder(root=data_dir, transform=preprocess)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
torch.manual_seed(42) # 분할 일관성을 위해 고정
_, test_db = torch.utils.data.random_split(full_dataset, [train_size, test_size])
test_loader = DataLoader(test_db, batch_size=32, shuffle=False)

def evaluate_saved_model(model_name, model_type='resnet18'):
    print(f"🧐 {model_name} 평가 중...")
    
    # 모델 구조 생성
    if model_type == 'resnet18':
        model = models.resnet18()
    else:
        model = models.resnet34()
    
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # 저장된 가중치 불러오기
    model.load_state_dict(torch.load(f"{model_name}.pth"))
    model = model.to(device)
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 지표 계산
    acc = (torch.tensor(all_labels) == torch.tensor(all_preds)).float().mean().item()
    pre = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return {"Acc": acc, "Pre": pre, "Rec": rec}

# 결과 수집
final_metrics = {}
final_metrics['Exp1'] = evaluate_saved_model('Exp1', 'resnet18')
final_metrics['Exp2'] = evaluate_saved_model('Exp2', 'resnet18')
final_metrics['Exp3'] = evaluate_saved_model('Exp3', 'resnet18')
final_metrics['Exp4'] = evaluate_saved_model('Exp4', 'resnet18')
final_metrics['Exp5'] = evaluate_saved_model('Exp5', 'resnet34')

# 표 형태로 출력
print("\n" + "="*50)
print(f"{'Experiment':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
print("-" * 50)
for name, m in final_metrics.items():
    print(f"{name:<10} | {m['Acc']*100:>8.2f}% | {m['Pre']:>10.4f} | {m['Rec']:>10.4f}")
print("="*50)