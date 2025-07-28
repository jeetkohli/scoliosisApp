import streamlit as st
from PIL import Image
from cobbAngleInput import analyze_image


st.title("Scoliosis Cobb Angle Analyzer")


uploaded_file = st.file_uploader("Upload spine X-ray", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image, caption="Uploaded Image", use_container_width=True)


	with st.spinner("Analyzing..."):
		angle, result = analyze_image(uploaded_file)


	if angle is not None:
		st.success(f"Cobb Angle: {angle}° — Severity: {result}")
	else:
		st.error(result)




