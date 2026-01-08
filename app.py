import streamlit as st
from transformers import pipeline
from PIL import Image
import plotly.express as px
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="이미지 분류 AI Application", 
    layout="centered",
    page_icon="👩‍🔬✨"
    )

# 모델 로딩 (캐싱)
# 이렇게 해놔야 매번 새로고침할 때마다 모델을 다시 로드하지 않음
@st.cache_resource
def load_model():
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    return classifier

def get_emoji(label):
    label_lower = label.lower()

    # 동물
    if any(word in label_lower for word in ['dog', 'puppy', 'pug', 'corgi', 'retriever']):
        return '🐶'
    elif any(word in label_lower for word in ['cat', 'kitten', 'tabby']):
        return '🐱'
    elif any(word in label_lower for word in ['bird', 'parrot', 'eagle', 'owl']):
        return '🦅'
    elif any(word in label_lower for word in ['fish', 'goldfish', 'shark']):
        return '🐟'
    elif any(word in label_lower for word in ['bear', 'panda']):
        return '🐻'
    elif any(word in label_lower for word in ['elephant']):
        return '🐘'
    elif any(word in label_lower for word in ['monkey', 'ape', 'gorilla']):
        return '🐵'
    
    # 음식
    elif any(word in label_lower for word in ['pizza', 'burger', 'sandwich', 'hot dog', 'taco']):
        return '🍕'
    elif any(word in label_lower for word in ['cake', 'cupcake', 'dessert', 'ice cream']):
        return '🍰'
    elif any(word in label_lower for word in ['coffee', 'espresso', 'latte']):
        return '☕'
    elif any(word in label_lower for word in ['beer', 'wine', 'cocktail']):
        return '🍺'
    
    # 차량
    elif any(word in label_lower for word in ['car', 'sports car', 'convertible', 'racer']):
        return '🚗'
    elif any(word in label_lower for word in ['truck', 'pickup']):
        return '🚚'
    elif any(word in label_lower for word in ['bus', 'school bus']):
        return '🚌'
    elif any(word in label_lower for word in ['plane', 'airliner', 'aircraft']):
        return '✈️'
    elif any(word in label_lower for word in ['boat', 'ship', 'vessel']):
        return '🚢'
    
    # 의류
    elif any(word in label_lower for word in ['suit', 'tie', 'gown', 'dress']):
        return '👔'
    elif any(word in label_lower for word in ['shoe', 'sneaker', 'boot']):
        return '👟'
    
    # 자연
    elif any(word in label_lower for word in ['flower', 'rose', 'daisy']):
        return '🌸'
    elif any(word in label_lower for word in ['tree', 'plant']):
        return '🌳'
    
    # 기본값
    else:
        return '🎯'

# 메인 타이틀
st.title("👩‍🔬✨ 이미지 분류 AI")
st.write("이미지를 업로드하면 어떤 이미지인지 알려드려요!🥨❣")

# 모델 로드
classifier = load_model()

# 파일 업로더
uploaded_file = st.file_uploader(
    "이미지를 업로드해주세요! 💌", 
    type=["png", "jpg", "jpeg"]
    )

if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:  # 가운데 컬럼에 이미지 배치
        st.image(image, caption="업로드한 이미지", use_container_width=True)

    # 분류 버튼
    if st.button("🔍 이미지 분류하기", type="primary"):
        with st.spinner("이미지 분류 중... 잠시만 기다려주세요! ⏳"):
            results = classifier(image, top_k=5)

        # 결과 출력
        st.subheader("📊 분류 결과")

        # Top 1 결과 강조
        top_result = results[0]
        emoji = get_emoji(top_result['label'])
        st.success(f"**{emoji} **({top_result['score']*100:.2f}%)")

        # 상위 5개 결과를 DataFrame으로 변환
        st.write("---")
        st.write("**상위 5개 예측 시각화:**")
        
        # 데이터 준비
        df = pd.DataFrame(results)
        df['score_percent'] = df['score'] * 100 # 백분율로 변환하기 (변수 이름 수정했음)

        # Plotly 막대 차트 생성
        fig = px.bar(
            df,
            x='score_percent',
            y='label',
            orientation='h',
            labels={'score_percent': '확률: (%)', 'label': '분류'},
            title='Top 5 예측 결과',
            color='score_percent',
            color_continuous_scale='Blues'
        )

        # 차트 레이아웃 조정
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            height=400
        )

        # 차트 표시
        st.plotly_chart(fig, use_container_width=True)

        # 상세 수치 표시
        st.write("---")
        st.write("**상세 결과:**")
        for i, result in enumerate(results, 1):
            emoji = get_emoji(result['label'])
            st.write(f"{i}. {emoji} {result['label']}: {result['score']*100:.2f}%")
