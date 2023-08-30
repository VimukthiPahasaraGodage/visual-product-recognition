import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Title of the app
st.title("Visual Product Recognition Demo")

# Adding text
st.write("Visual Product Recognition systems let you find the product you are looking for very easily by replacing the"
         " traditional text queries with image queries. Let's get started...")
image_data = st.file_uploader(label='Upload the image you want to query on', type=['jpg', 'png'])

canvas_max_height = 500
canvas_max_width = 500

if image_data is not None:
    uploaded_image = Image.open(image_data)
    width, height = uploaded_image.size
    st.write(f"{width}, {height}")

    canvas_height = canvas_max_height
    canvas_width = canvas_max_width

    if width >= height:
        image_pixel_by = canvas_max_width / width
        canvas_height = int(height * image_pixel_by)
    else:
        image_pixel_by = canvas_max_width / width
        canvas_width = int(width * image_pixel_by)

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # Fixed fill color with some opacity
        stroke_width=2,
        stroke_color="#FFFF00",
        background_color="#EEEEEE",
        background_image=uploaded_image,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="rect",
        key="canvas"
    )

# drawing_mode = st.sidebar.selectbox(
#     "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
# )
#
# stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
# if drawing_mode == 'point':
#     point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
# stroke_color = st.sidebar.color_picker("Stroke color hex: ")
# bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

# canvas_result = st_canvas(
#     fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
#     stroke_width=5,
#     stroke_color="#00FFFF",
#     background_color="#DDDDDD",
#     background_image=Image.open(bg_image) if bg_image else None,
#     update_streamlit=realtime_update,
#     height=150,
#     drawing_mode=drawing_mode,
#     point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
#     key="canvas",
# )

# realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Adding a plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
st.pyplot(plt)
