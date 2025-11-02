import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# -------------------------------
# Enhancement functions
# -------------------------------

def enhance_landscape(img, brightness=10, contrast=1.1, sharpen_intensity=0.1):
    """Enhance landscape with adjustable settings."""
    img_cv = np.array(img)

    # LAB conversion for brightness/contrast
    lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE for subtle contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8,8))
    l = clahe.apply(l + brightness)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Gentle sharpening
    kernel = np.array([[0, -sharpen_intensity, 0],
                       [-sharpen_intensity, 1 + sharpen_intensity*4, -sharpen_intensity],
                       [0, -sharpen_intensity, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Slight vibrance boost
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 5)
    hsv = cv2.merge((h, s, v))
    final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return final

def enhance_portrait(img, faces=None, brightness=10, contrast=1.05, sharpen_intensity=0.1, smooth_intensity=5):
    """Enhance portrait with adjustable settings."""
    img_cv = np.array(img)

    # Subtle brightness and saturation adjustment
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, brightness)
    s = cv2.add(s, 5)
    hsv = cv2.merge((h, s, v))
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Minimal sharpening
    kernel = np.array([[0, -sharpen_intensity, 0],
                       [-sharpen_intensity, 1 + sharpen_intensity*4, -sharpen_intensity],
                       [0, -sharpen_intensity, 0]])
    sharpened = cv2.filter2D(adjusted, -1, kernel)

    # Gentle smoothing for face regions
    if faces:
        for (x, y, w, h) in faces:
            face_region = sharpened[y:y+h, x:x+w]
            smoothed = cv2.bilateralFilter(face_region, d=smooth_intensity, sigmaColor=50, sigmaSpace=50)
            sharpened[y:y+h, x:x+w] = smoothed

    return sharpened

# -------------------------------
# Category detection
# -------------------------------

def detect_category(img):
    """Detect portrait, landscape, or mixed based on face area and sky pixels."""
    img_cv = np.array(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detected_faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=8, minSize=(50,50)
    )

    # Filter significant faces (>1% of image area)
    faces = [(x, y, w, h) for (x, y, w, h) in detected_faces if w*h > 0.01*gray.shape[0]*gray.shape[1]]

    # Simple landscape heuristic (sky pixels in top third)
    height, width = gray.shape
    top = img_cv[:height//3]
    blue_pixels = np.sum((top[:,:,2] < 100) & (top[:,:,0] > 100))
    landscape = blue_pixels > 5000

    if len(faces) > 0 and landscape:
        return "mixed", faces
    elif len(faces) > 0:
        return "portrait", faces
    elif landscape:
        return "landscape", []
    else:
        return "unknown", []

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="SnapLab", layout="wide")
st.title("SnapLab - DSLR-Style Auto Image Enhancer")

uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg","jpeg","png"])

# User sliders for adjustments
st.sidebar.header("Enhancement Controls")
brightness = st.sidebar.slider("Brightness", -20, 50, 10)
contrast = st.sidebar.slider("Contrast (CLAHE clipLimit)", 1.0, 3.0, 1.2, 0.1)
sharpen_intensity = st.sidebar.slider("Sharpening", 0.0, 0.5, 0.1, 0.05)
smooth_intensity = st.sidebar.slider("Face Smoothness", 1, 15, 5)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    category, faces = detect_category(image)
    st.write(f"Detected category: **{category}**")

    if category == "landscape":
        enhanced = enhance_landscape(image, brightness, contrast, sharpen_intensity)
    elif category == "portrait":
        enhanced = enhance_portrait(image, faces, brightness, contrast, sharpen_intensity, smooth_intensity)
    elif category == "mixed":
        # Apply landscape enhancement first
        land_enhanced = enhance_landscape(image, brightness, contrast, sharpen_intensity)
        # Portrait enhancement only on faces
        enhanced = enhance_portrait(land_enhanced, faces, brightness, contrast, sharpen_intensity, smooth_intensity)
    else:
        st.write("No enhancements applied.")
        enhanced = np.array(image)

    st.image(enhanced, caption="Enhanced Image", use_column_width=True)

    # Download button
    buf = BytesIO()
    Image.fromarray(enhanced).save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download Enhanced Image",
                       data=byte_im,
                       file_name="enhanced.png",
                       mime="image/png")
