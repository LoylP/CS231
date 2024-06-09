import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image
import joblib
from skimage.feature import hog
import base64

st.set_page_config(
    page_title='Traffic Sign Classifier',
    page_icon='🚦',
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main { background-color: #000015; }
        .stButton>button { color: white; background-color: #4CAF50; }
        .stSidebar .css-1d391kg { background-color: #003366; }
        .css-17z5llp { color: white; }
    </style>
""", unsafe_allow_html=True)

# Title
st.title(':green[Classify 3 types of traffic signs]')

# Sidebar menu
st.sidebar.title("Menu")
uploaded_img = st.sidebar.file_uploader('Upload a PNG or JPG image', type=['jpg', 'png'])
algorithm = st.sidebar.selectbox(
    'Select the algorithm you want',
    ('Choose', 'Decision Tree Classification', 'KNeighborsClassifier', 'SVM')
)
display_size = st.sidebar.slider(
    'Select the display size for prediction label',
    min_value=20, max_value=100, value=50
)
st.sidebar.title("About")
st.sidebar.info("This application classifies traffic signs into three categories: Danger signs, Prohibition signs, and Announcement signs.")

def display_prediction(pred):
        label = "predicted results: "
        if pred[0] == 0.0:
            label = "Not a traffic sign"
        elif pred[0] == 1.0:
            label = "Danger signs"
        elif pred[0] == 2.0:
            label = "Prohibition signs"
        elif pred[0] == 3.0:
            label = "Announcement sign"
        
        st.markdown(f"<h1 style='font-size:{display_size}px;'>{label}</h1>", unsafe_allow_html=True)
        st.write(pred[0])
        return label

def test_prediction(img_array, svc, size, pixel):
            img_gray = cv2.cvtColor(cv2.resize(img_array, size), cv2.COLOR_RGB2GRAY)
            image, viz = hog(img_gray, orientations=9, pixels_per_cell=pixel,
                             cells_per_block=(2, 2), visualize=True)

            x_tst = np.asarray(image)
            pred = svc.predict([x_tst])
            label = display_prediction(pred)
            return label

load = False
if uploaded_img is not None:
    image = Image.open(uploaded_img)
    img_array = np.array(image)
    st.image(image, caption="Uploaded Image")
    load = True

if load:
    st.header(f":white[Algorithm: {algorithm}]")

    if algorithm == 'Choose':
        st.title(':red[Select the algorithm you want ]')

    elif algorithm == 'SVM':
        with open('svm_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        if st.button('Predict'):
            st.title(":green[Prediction Result: ]")
            result_label = test_prediction(img_array, loaded_model, (128, 128), (8, 8))


    elif algorithm == 'KNeighborsClassifier':
        with open('knn_model.pkl', 'rb') as file:
            loaded_model = joblib.load(file) 

        if st.button('Predict'):
            st.title(":green[Prediction Result: ]")
            result_label = test_prediction(img_array, loaded_model, (128, 128), (8, 8))

    else:  # Decision Tree Classification
        with open('decision_tree.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        if st.button('Predict'):
            st.title(":green[Prediction Result: ]")
            result_label = test_prediction(img_array, loaded_model, (128, 128), (8, 8))
           