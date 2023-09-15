import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import datetime
import json
from streamlit_lottie import st_lottie
st.set_page_config(page_title='FACE MASK DETECTION SYSTEM',page_icon='random')
st.sidebar.image("https://tse3.mm.bing.net/th?id=OIP.bgQyu98mv5R51Ow4PllbkwHaEu&pid=Api&P=0&h=180")
choice=st.sidebar.selectbox("Menu",("Home","URL IP CAMERA","CAMERA"))
st.header(choice)
if(choice=="Home"):
    st.markdown("<h1 style= 'text-align:center;color:brown;'>FACE MASK DETECTION SYSTEM</h1>",unsafe_allow_html=True)
    st.image("https://www.ideas2it.com/wp-content/uploads/2020/05/Facemask-Detection-Blog.jpg")
    with open("animation_lmjhllw1.json") as source:
        animation=json.load(source)
    st_lottie(animation)
elif(choice=="URL IP CAMERA"):
    st.image("https://sunnybellary.com/project/facemaskdetector/ftr_hua13f70f7f508df814fa8aa9bfed215bc_127650_2000x2000_fit_q90_lanczos.jpg")
    url=st.text_input("Enter your URL")
    btn= st.button("Start Detection")
    window=st.empty()
    if btn:
        i=1;
        btn2=st.button("Stop Detection")
        if btn2:
            st.experimental_rerun()
        facemodel=cv2.CascadeClassifier("face.xml")
        maskmodel=load_model("mask.h5")
        vid=cv2.VideoCapture(url)
        while(vid.isOpened()):
            flag,frame=vid.read()
            if flag:
                faces=facemodel.detectMultiScale(frame)
                for(x,y,l,w) in faces:
                    face_img=frame[y:y+w,x:x+l]
                    face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
                    face_img=np.asarray(face_img, dtype=np.float32).reshape(1,224,224,3)
                    face_img=(face_img/127.5)-1
                    p=maskmodel.predict(face_img)[0][0]
                    if(p>0.9):
                        path="NoMask/"+str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                window.image(frame,channels='BGR')
                
elif(choice=="CAMERA"):
    st.image("https://proactive.co.in/wp-content/uploads/2020/08/face-mask-detection.jpg")
    cam=st.selectbox("Choose Camera",("None","Primary", "Secondary"))
    btn= st.button("Start Detection")
    window=st.empty()
    if btn:
        i=1;
        btn2=st.button("Stop Detection")
        if btn2:
            st.experimental_rerun()
        facemodel=cv2.CascadeClassifier("face.xml")
        maskmodel=load_model("mask.h5")
        if cam=="Primary":
            cam=0
        else:
            cam=1
        vid=cv2.VideoCapture(cam)
        while(vid.isOpened()):
            flag,frame=vid.read()
            if flag:
                faces=facemodel.detectMultiScale(frame)
                for(x,y,l,w) in faces:
                    face_img=frame[y:y+w,x:x+l]
                    face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
                    face_img=np.asarray(face_img, dtype=np.float32).reshape(1,224,224,3)
                    face_img=(face_img/127.5)-1
                    p=maskmodel.predict(face_img)[0][0]
                    if(p>0.9):
                        path="NoMask/"+str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+w,x:x+l])
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                window.image(frame,channels='BGR')
