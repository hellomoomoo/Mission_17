import streamlit as st
from transformers import pipeline
from PIL import Image
import plotly.express as px
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="이미지 분류 AI Application", # 브라우저 탭에 표시될 제목
    layout="centered",      # 페이지 레이아웃 / centered는 중앙 정렬
    page_icon="👩‍🔬✨"       # 웹 아이콘
)

# 모델 로딩 (캐싱)
@st.cache_resource  # 함수 결과를 캐시에 저장
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


# 모델 로드
classifier = load_model()   # 웹 시작시에 한 번만 로드됨

# 화면 비율은 1:1 비율로 2개 컬럼으로 나눔
left_col, right_col = st.columns([1, 1])

# 왼쪽 컬럼: 입력 및 업로드
with left_col:
    # 타이틀
    st.title("👩‍🔬✨ 이미지 분류 AI")
    st.write("이미지를 업로드하면 어떤 이미지인지 알려드려요!🥨❣")

    # 입력 방식 선택
    st.subheader("📸 이미지 입력 방법 선택")

    input_method = st.radio(
    "어떤 방식으로 이미지를 업로드하시겠어요?",
    ["📁 파일 업로드", "📷 카메라 촬영"],
    horizontal=True # 옵션을 가로로 배치 
    )

    # 업로드된 이미지를 담을 빈 리스트
    images = []

    # 파일 업로드 방식 (단일 파일이었다가 다중 파일로 변경)
    if input_method == "📁 파일 업로드":
        uploaded_files = st.file_uploader(
            "이미지를 업로드해주세요! (여러 장도 가능해요) 💌", 
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True  # 여러 파일 동시 업로드 허용
        )

        # 업로드된 파일들을 PIL 이미지로 변환하여 리스트에 저장
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                images.append(Image.open(uploaded_file))

    # 카메라 촬영 방식
    else:
        camera_photo = st.camera_input("카메라로 사진을 찍어주세요! 📸")
        if camera_photo is not None:
            images.append(Image.open(camera_photo))


# 오른쪽 컬럼: 이미지 미리보기

# 이미지가 있으면 (업로드 or 촬영)
if len(images) > 0:
    st.markdown("### 📷 업로드된 이미지")
    st.write(f"총 {len(images)}장의 이미지가 업로드되었습니다!")

    # 각 이미지를 순서대로 표시
    for idx, image in enumerate(images, 1):
        # enumerate(images, 1): 인덱스를 1부터 시작 (0 아님)
        st.image(
            image,
            caption=f"이미지 {idx}",    # 이미지 아래 캡션
            width=500 # 고정 너비 (픽셀 단위)
        )
        if idx < len(images): 
            st.write("---")  # 이미지 사이에 구분선 추가

        

if len(images) > 0:
    # 분류 버튼 (중앙 정렬)
    col1, col2, col3 = st.columns([1, 2, 1])    # 1:2:1 비율로 3개 컬럼 생성
    with col2:
        classify_button = st.button(
            "🔍 모든 이미지 분류하기", 
            type="primary", # 파란색 강조 버튼
            use_container_width=True    # 컬럼 너비에 맞춤
        )

    # 버튼이 클릭되면 분류 시작    
    if classify_button:

        # 각 이미지마다 분류
        for idx, image in enumerate(images, 1):
            st.write(f"이미지 {idx}")

            # 이미지 표시
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption=f"이미지 {idx}", width=500)

            # 분류 수행
            with st.spinner(f"이미지 {idx} 분류 중"):
                results = classifier(image, top_k=5)

            # 결과 출력
            st.subheader("📊 **분류 결과:**")

            # Top 1 결과 강조 (이모지 추가 + Label 추가)
            top_result = results[0]
            emoji = get_emoji(top_result['label'])
            st.success(f"{emoji} **{top_result['label']}** ({top_result['score']*100:.2f}%)")

            # Plotly 차트
            df = pd.DataFrame(results)
            df['score_percent'] = df['score'] * 100
            
            fig = px.bar(
                df,
                x='score_percent',
                y='label',
                orientation='h',
                labels={'score_percent': '확률 (%)', 'label': '분류'},
                title=f'이미지 {idx} - Top 5 예측 결과',
                color='score_percent',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=300  # 여러 개니까 높이 줄임
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 상세 결과
            with st.expander("상세 결과 보기"):
                for i, result in enumerate(results, 1):
                    emoji = get_emoji(result['label'])
                    st.write(f"{i}. {emoji} {result['label']}: {result['score']*100:.2f}%")