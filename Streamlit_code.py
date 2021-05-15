import streamlit as st #for webapp
import NLP_model as model

#designing the web app
st.title('Spam Classifier')
st.image('image.jpg')
user_input=st.text_input('Write your message here.')
submit=st.button('Predict')
if submit:
    answer=model.predict([user_input])
    st.text(answer)