import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import json
import os

# File to store user credentials
USER_DB_FILE = 'user_db.json'

def load_user_db():
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_db(user_db):
    with open(USER_DB_FILE, 'w') as f:
        json.dump(user_db, f)

def main_page():
    st.title('Object Detection Web App')
    st.subheader('Powered by YOLOv8')
    st.write(f'Welcome, {st.session_state.username}!')
    
    model = YOLO('./Yolomodelpredictedwieght.pt')
    object_names = list(model.names.values())

    color_map = {
        'car': (0, 255, 0),
        'smoke': (0, 0, 255),
        'default': (255, 0, 0)
    }

    with st.form("my_form"):
        file_type = st.radio("Choose file type", ('Image', 'Video'))
        uploaded_file = st.file_uploader("Upload file", type=['jpg', 'jpeg', 'png', 'mp4'])
        selected_objects = st.multiselect('Choose objects to detect', object_names, default=[obj for obj in ['car', 'smoke'] if obj in object_names])
        min_confidence = st.slider('Confidence score', 0.0, 1.0)
        submit_button = st.form_submit_button(label='Submit')

    if submit_button and uploaded_file is not None:
        if file_type == 'Image':
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            results = model(image_np)

            with st.spinner('Processing image...'):
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x0, y0 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
                        x1, y1 = int(box.xyxy[0][2]), int(box.xyxy[0][3])
                        score = round(float(box.conf[0]), 2)
                        cls = int(box.cls[0])
                        object_name = model.names[cls]
                        label = f'{object_name} {score}'

                        color = color_map.get(object_name, color_map['default'])

                        if object_name in selected_objects and score > min_confidence:
                            cv2.rectangle(image_np, (x0, y0), (x1, y1), color, 2)
                            cv2.putText(image_np, label, (x0, y0 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption='Original Image', use_column_width=True)
                with col2:
                    st.image(image_np, caption='Processed Image', use_column_width=True)

        elif file_type == 'Video':
            input_path = uploaded_file.name
            file_binary = uploaded_file.read()
            with open(input_path, "wb") as temp_file:
                temp_file.write(file_binary)
            video_stream = cv2.VideoCapture(input_path)
            width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'h264')
            fps = int(video_stream.get(cv2.CAP_PROP_FPS))
            output_path = input_path.split('.')[0] + '_output.mp4'
            out_video = cv2.VideoWriter(output_path, int(fourcc), fps, (width, height))

            with st.spinner('Processing video...'):
                while True:
                    ret, frame = video_stream.read()
                    if not ret:
                        break
                    results = model(frame)
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x0, y0 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
                            x1, y1 = int(box.xyxy[0][2]), int(box.xyxy[0][3])
                            score = round(float(box.conf[0]), 2)
                            cls = int(box.cls[0])
                            object_name = model.names[cls]
                            label = f'{object_name} {score}'

                            color = color_map.get(object_name, color_map['default'])

                            if object_name in selected_objects and score > min_confidence:
                                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                                cv2.putText(frame, label, (x0, y0 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    out_video.write(frame)
                video_stream.release()
                out_video.release()

            col1, col2 = st.columns(2)
            with col1:
                st.video(input_path, format="video/mp4")
            with col2:
                st.video(output_path, format="video/mp4")

def login_page():
    st.title('Login Page')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    
    if st.button('Login'):
        user_db = load_user_db()
        if username in user_db and user_db[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.page = 'main'
        else:
            st.error('Invalid username or password.')

def register_page():
    st.title('Register Page')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    
    if st.button('Register'):
        user_db = load_user_db()
        if username in user_db:
            st.error('Username already exists.')
        else:
            user_db[username] = password
            save_user_db(user_db)
            st.session_state.username = username
            st.session_state.logged_in = True
            st.session_state.page = 'main'

def initial_page():
    st.title('Welcome to the Object Detection App')
    option = st.selectbox('Are you a new user or an existing user?', ['New User', 'Existing User'])
    
    if option == 'New User':
        st.write('Please register to create an account.')
        register_page()
    elif option == 'Existing User':
        st.write('Please log in to access the app.')
        login_page()

def app():
    # Initialize session state if not already set
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.page = 'initial'

    # Display appropriate page based on session state
    if st.session_state.logged_in:
        main_page()
    else:
        if st.session_state.page == 'initial':
            initial_page()
        elif st.session_state.page == 'login':
            login_page()
        elif st.session_state.page == 'register':
            register_page()

if __name__ == "__main__":
    app()
