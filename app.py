import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# [수정 1] 가장 먼저 호출되어야 함!
st.set_page_config(page_title="Pokemon Classifier", page_icon="⚡")
# 1. 환경 설정 및 클래스 이름 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = os.path.dirname(__file__)
data_dir = os.path.join(base_path, 'PokemonData')

if os.path.exists(data_dir):
    class_names = sorted(os.listdir(data_dir))
else:
    # 폴더를 못 찾을 경우를 대비해 임시 메시지 (에러 방지)
    st.warning(f"경로를 확인해주세요: {os.path.abspath(data_dir)}")
    class_names = [f"Class_{i}" for i in range(150)]

# 2. 모델 로드 함수 (Exp4: ResNet18 기반)
@st.experimental_singleton
def load_pokemon_model():
    # Exp4는 ResNet18 구조를 사용함
    model = models.resnet18()
    num_classes = len(class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # [수정] 실행 중인 app.py 파일의 폴더 경로를 구해서 파일명과 합칩니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "Exp4.pth")
    
    # 경로가 진짜 맞는지 터미널에 출력해서 확인용
    # print(f"모델을 찾는 경로: {model_path}") 
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

# 3. 이미지 전처리 설정 (학습 시 사용한 weights.transforms()와 동일하게 구성)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- UI 부분 ---
st.title("🛡️ 포켓몬 분류기")
st.write("이미지를 업로드하면 포켓몬 이름을 맞춥니다!")

uploaded_file = st.file_uploader("포켓몬 이미지를 선택하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='내가 업로드한 포켓몬', use_column_width=True)
    
    # 분류 진행
    with st.spinner('분석 중...'):
        model = load_pokemon_model()
        
        # 전처리 및 추론
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # 상위 5개 결과 추출[cite: 1]
            top5_prob, top5_idx = torch.topk(probabilities, 5)

    st.success("분석 완료!")
    st.subheader("Top-5 예측 결과")
    
    # 결과 출력[cite: 1]
    for i in range(5):
        name = class_names[top5_idx[i]]
        prob = top5_prob[i].item() * 100
        st.write(f"**{i+1}위: {name}** ({prob:.2f}%)")
        st.progress(prob / 100)