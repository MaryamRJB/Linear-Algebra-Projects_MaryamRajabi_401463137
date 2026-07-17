import os
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from svd_utils import compress_image, calculate_psnr

st.set_page_config(
    page_title="SVD Image Compression",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Compression using SVD")

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    img = np.array(image)

    original_size = len(uploaded_file.getvalue()) / 1024

    st.subheader("Image Information")

    st.write(f"Filename : {uploaded_file.name}")
    st.write(f"Dimensions : {image.width} × {image.height}")
    st.write(f"Original Size : {original_size:.2f} KB")

    st.markdown("---")

    method = st.radio(

        "Compression Method",

        [

            "Number of Components",

            "Compression Percentage"

        ]

    )

    max_k = min(image.width, image.height)

    if method == "Number of Components":

        k = st.slider(

            "Select k",

            1,

            max_k,

            50

        )

    else:

        percent = st.slider(

            "Compression Percentage",

            1,

            100,

            50

        )

        k = max(1, int(max_k * percent / 100))

        st.write(f"Calculated Components : {k}")

    if st.button("Compress Image"):

        start = time.time()

        compressed = compress_image(img, k)

        execution = time.time() - start

        compressed_image = Image.fromarray(compressed)

        os.makedirs("outputs", exist_ok=True)

        save_path = "outputs/compressed_image.png"

        compressed_image.save(save_path)

        psnr = calculate_psnr(img, compressed)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original")

            st.image(image, use_container_width=True)

        with col2:

            st.subheader("Compressed")

            st.image(compressed_image, use_container_width=True)

        st.markdown("---")

        st.subheader("Results")

        st.write(f"Original Size : {original_size:.2f} KB")

        st.write(f"Components Used : {k}")

        st.write(f"Execution Time : {execution:.4f} sec")

        st.write(f"PSNR : {psnr:.2f} dB")

        rows, cols = img.shape[:2]

        original_values = rows * cols * 3
        compressed_values = k * (rows + cols + 1) * 3

        ratio = original_values / compressed_values

        st.write(f"Compression Ratio : {ratio:.2f} : 1")



        with open(save_path, "rb") as f:

            st.download_button(

                "⬇ Download Image",

                f,

                file_name="compressed_image.png"

            )

        st.markdown("---")

        st.subheader("Quality Comparison")

        sample = [5, 10, 20, 40, 60, 80, 100]

        qualities = []

        for value in sample:

            kk = max(1, int(max_k * value / 100))

            temp = compress_image(img, kk)

            qualities.append(calculate_psnr(img, temp))

        fig, ax = plt.subplots(figsize=(8,4))

        ax.plot(sample, qualities, marker="o")

        ax.set_xlabel("Percentage of Components")

        ax.set_ylabel("PSNR")

        ax.set_title("Compression Quality")

        st.pyplot(fig)