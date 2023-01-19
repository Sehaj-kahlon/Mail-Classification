import streamlit as st
import pickle
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
ps = PorterStemmer()
#this function can also be exported using pickle
def transform_text(text):
    text = text.lower() #convert to lowercase
    text = nltk.word_tokenize(text) #create a list of all the words

    y = [] #removing special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i  not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    for i in text:
         y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Add css to make text bigger
st.markdown(
    """
    <style>
    textarea {
        font-size: 1.5rem !important;
    }
    input {
        font-size: 3rem !important;
    }
    label{
        font-size: 89rem !important;
    }
    .css-1yy6isu p {
    word-break: break-word;
    font-size: 40px;
    }
    p, ol, ul, dl {
    margin: 0px 0px 1rem;
    padding: 0px;
    font-size: 1.5rem;
    font-weight: 400;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Email/SMS Spam Classifier")
# original_title = '<p style="font-family:Sans serif; font-size: 40px;">Enter the Message</p>'
# st.markdown(original_title, unsafe_allow_html=True)
input_sms = st.text_area(label = "Enter the Message" )

if st.button('Predict'):
# 1. preprocessing 2. vectorize 3. predict 4. display
    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]


    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
