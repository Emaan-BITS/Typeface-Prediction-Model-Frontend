import streamlit as st
from streamlit_cropper import st_cropper

import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from model.model import SimbleModel

MODEL_PATH = "model/e29 b00324 - l 0.94693 (complete).pth"
FONTNAMES_PATH = "model/fontnames all.txt"

st.set_page_config(
  page_title="DL-based Typeface Recognition",
)

st.title("DL-based Typeface Recognition")
st.write("""
The app allows users to upload an image of a character (screenshot, camera
capture, etc.), and it will predict the typeface used in the image. The model has
been trained on a diverse set of typefaces to ensure accurate recognition.
  """
)

# region Inputs
if "model" not in st.session_state:
  st.warning("Model status: loading, please wait a moment...", icon="⏳")

else:
  st.success("Model status: loaded!", icon="✅")

  # MARK: | Crop dialog
  @st.dialog("🖼️ Crop the image to the character", width="large")
  def crop(img: Image.Image):

    st.write("Use the zoom slider to enlarge the image for your convenience.")
    SCALE = st.slider("Zoom", 1, 5, 2)

    st.write("Adjust the following so that the resulting image is **black text on white background.**")
    CONTRAST = st.slider("Contrast", 0.5, 5.0, 1.0)
    BRIGHTNESS = st.slider("Brightness", 0.0, 2.0, 1.0)

    # THRESHOLD_ACTIVATE = st.checkbox(label="Enable mono threshold?")
    # THRESHOLD_SLIDER = st.slider("Mono threshold", min_value=0, max_value=255, value=127, disabled=not THRESHOLD_ACTIVATE)

    box = st_cropper(
      img.resize((img.width * SCALE, img.height * SCALE)),
      realtime_update=True,
      aspect_ratio=None,
      should_resize_image=True,
      return_type="box",
    )

    # left, upper, right, lower
    final_image = img.crop(
      (
        (box["left"]) / SCALE,
        (box["top"]) / SCALE,
        (box["left"] + box["width"]) / SCALE,
        (box["top"] + box["height"]) / SCALE,
      )
    )

    final_image = final_image.convert("L")
    final_image = ImageEnhance.Contrast(final_image).enhance(CONTRAST)
    final_image = ImageEnhance.Brightness(final_image).enhance(BRIGHTNESS)
    # if THRESHOLD_ACTIVATE:
    #     margin = 20

    #     map = list(range(256))
    #     map[:max(0, THRESHOLD_SLIDER-margin)] = [0] * max(0, THRESHOLD_SLIDER-margin)
    #     map[max(0, THRESHOLD_SLIDER-margin):min(THRESHOLD_SLIDER+margin, 255)] = np.linspace(0, 255, 2*margin, endpoint=True, dtype=np.uint8)
    #     map[min(THRESHOLD_SLIDER+margin, 255):] = [255] * (256 - min(THRESHOLD_SLIDER+margin, 255))

    #     final_image = final_image.point(map, mode="L")

    inv = st.checkbox("Invert image?", False)
    if inv: final_image = ImageOps.invert(final_image)

    st.write("Preview:")
    st.image(final_image)
    st.write("Ensure the above image is **Black text on White background**")
    if st.button("Save Crop"):
      st.session_state.img = final_image
      st.rerun()

  # MARK: | Upload and Character Input
  with st.container():
    col1, col2 = st.columns(2)

    with col1:
      with st.container():
        font_image = st.file_uploader(
          "📤 Upload Image",
          type=["jpg", "png", "jpeg"],
          accept_multiple_files=False,
        )

        if font_image:
          if "img" not in st.session_state:
            st.session_state.img = None
            img = Image.open(font_image)
            crop(img)
          else:
            st.image(
              st.session_state.img,
            )
            if st.button("Change Image"):
              st.session_state.pop("img")
              img = Image.open(font_image)
              crop(img)
        else:
          if "img" in st.session_state:
            st.session_state.pop("img")
          st.error("⚠️ Please upload an image")

    with col2:
      character = st.text_input(
        "🅱️ Enter Character",
        max_chars=1,
        placeholder="B",
      )

    predict_button = st.button(
      "Predict",
      width="stretch",
      disabled="img" not in st.session_state or not character,
    )

  # MARK: | Prediction and output
  if predict_button:
    if "model" not in st.session_state:
      st.error("⚠️ The model is not loaded yet, please try again.")

    else:
      st.divider()

      st.header("📊 Output")

      output = [
        {"Typeface": tf_name, "Confidence": score}
        for tf_name, score in st.session_state.model.predict(
          st.session_state.img, ord(character)
        )
      ]

      prediction = pd.DataFrame(output)

      st.dataframe(
        prediction,
        width="stretch",
        hide_index=True,
        column_config={
          "Confidence": st.column_config.ProgressColumn(width="large")
        },
      )

      top_typeface = output[0]["Typeface"]
      st.success(f"Top typeface: {top_typeface}")

# endregion

st.divider()

# MARK: Info


st.header("ℹ️ Usage instructions")
st.write("""
1. Click the **"Upload Image"** button, and select an image file containing the
   text/character you want to analyze.
2. A cropping dialog will appear. Adjust the crop area to isolate **one**
   character you want to detect. Adjust brightness and contrast as necessary.
   When done, click **"Save Crop"**.
3. In the text input field, type the character. **Make sure it exactly matches**
   (case-sensitive), to ensure accurate predictions.
4. Click the **"Predict"** button. The app will display the predicted typeface
   in the **Output** section.""")

st.header("🖥️ More info")

st.subheader("About the Dataset")
st.write("""
The dataset used to train the model is synthetically generated. It is created
using 167 commonly available typefaces found on most modern Windows devices.

For each typeface, images are generated for the following characters:
- **Uppercase letters**: `A-Z`
- **Lowercase letters**: `a-z`
- **Digits**: `0-9`

Each image is preprocessed and then resized to a consistent size of **64x64
pixels**.""")

with st.expander("Typefaces used in the dataset"):
  typefaces_demo_df = pd.read_csv("assets/frontend_demo.csv")
  st.dataframe(
    typefaces_demo_df,
    hide_index=True,
    column_config={
      "Typeface name": st.column_config.Column(width="medium", pinned=True),
      "Sample text": st.column_config.ImageColumn(label=None, width="large", help=None),
    },
    width="stretch",
  )

st.subheader("About the Model")
st.write("""
The model is a **Convolutional Neural Network (CNN)** trained on the
above-described typeface dataset. CNNs are a type of Deep Learning models that are
particularly effective for image classification tasks."""
)

# Load the model after the webpage has been rendered
if "model" not in st.session_state:
  st.session_state["model"] = SimbleModel(MODEL_PATH, FONTNAMES_PATH)
  st.rerun()
