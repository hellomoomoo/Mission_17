import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(
    page_title="이미지 분류 AI Application", 
    layout="centered",
    page_icon="👩‍🔬✨"
    )

# 모델 로딩 (캐싱)
@st.cache_resource
def load_model():
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    return classifier

# 메인 타이틀
st.title("👩‍🔬✨ 이미지 분류 AI Application")
st.write("이미지를 업로드하면 어떤 이미지인지 알려드려요!🎞")

# 모델 로드
classifier = load_model()

# 파일 업로더
uploaded_file = st.file_uploader(
    "이미지를 업로드해주세요", 
    type=["png", "jpg", "jpeg"]
    )
if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", use_column_width=True)

    # 분류 버튼
    if st.button("이미지 분류하기", type="primary"):
        with st.spinner("이미지 분류 중... 잠시만 기다려주세요! ⏳"):
            results = classifier(image, top_k=5)

        # 결과 출력
        st.subheader("📊 분류 결과")

        # Top 1 결과 강조
        top_result = results[0]
        st.success(f"**{top_result['label']}** ({top_result['score']*100:.2f}%)")

        # 상위 5개 결과 출력
        st.write("---")
        st.write("**상위 5개 예측:**")
        for i, result in enumerate(results, 1):
            label = result['label']
            score = result['score']

            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{i}. {label}**")
            with col2:
                st.write(f"{score*100:.1f}%")

            st.progress(score)

