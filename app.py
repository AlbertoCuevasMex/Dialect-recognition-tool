import streamlit as st
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import spacy
import os

nlp = spacy.load("es_core_news_sm")
models_path = "model"


def testing ():

    model = st.selectbox("Select your model:", os.listdir(models_path))

    features_raw = pd.read_csv('testing/features.txt', sep=',', encoding='ISO-8859-1')
    features = features_raw['token'].tolist()

    test_eval = pd.read_csv('testing/test (eval).txt', sep='\t', encoding='utf-8')
    test_ar = test_eval[test_eval['Dialect_tag']=="AR"][0:7]
    test_ar["Options"] = "Argentinian --- " + test_ar['Content']
    test_mx = test_eval[test_eval['Dialect_tag']=="MX"][0:7]
    test_mx["Options"] = "Mexican --- " + test_mx['Content']
    test_sp = test_eval[test_eval['Dialect_tag']=="SP"][0:7]
    test_sp["Options"] = "Castilian --- " + test_sp['Content']

    test_df = pd.concat([test_ar, test_mx, test_sp], axis=0, ignore_index=True)


    option = st.selectbox("Select your sentence:", test_df['Options'].to_list())

    if st.button("Run"):
        loaded_model = pickle.load(open(f"model/{model}", "rb"))
        test_features = np.zeros((1, len(features)))
        
        content = test_df[test_df['Options']==option]["Content"].values[0]
        tag = test_df[test_df['Options']==option]["Dialect_tag"].values[0]

        test_doc_subs = nlp.pipe([content])
        for subs, feat in zip(test_doc_subs, test_features):
            tokens_list = [token.lower_ for token in subs]
            for term in features:
                if term in tokens_list:
                    term_id = features.index(term)
                    feat[term_id] = 1

        st.write(f"Sentence: {content}")
        st.write(f"Dialect Tag: {tag}")

        st.write(f"Prediction: {loaded_model.predict([test_features[0]])[0]}")

        df = pd.DataFrame(loaded_model.predict_proba([test_features[0]]), columns= loaded_model.classes_)
        st.write(df)

def one_sentence ():
    model = st.selectbox("Select your model:", os.listdir(models_path))

    sentence = st.text_input("Make your sentence to predict dialect tag:")
    st.write(f"Your sentence is: {sentence}")

    features_raw = pd.read_csv('testing/features.txt', sep=',', encoding='ISO-8859-1')
    features = features_raw['token'].tolist()

    if sentence =="":
        st.warning("Please add a sentence.")
    else:
        run = st.button("Run")

        if run:
            loaded_model = pickle.load(open(f"model/{model}", "rb"))
            test_features = np.zeros((1, len(features)))
            test_doc = nlp.pipe([sentence])
            for words, feat in zip(test_doc, test_features):
                tokens_list = [token.lower_ for token in words]
                for term in features:
                    if term in tokens_list:
                        term_id = features.index(term)
                        feat[term_id] = 1
            st.write(f"Sentence: {sentence}")
            st.write(f"Prediction Tag: {loaded_model.predict([test_features[0]])[0]}")
            df = pd.DataFrame(loaded_model.predict_proba([test_features[0]]), columns= loaded_model.classes_)
            st.write(df)

def main ():
    page = st.sidebar.selectbox("Pages", ["Make your prediction", "Examples", "Example Streamlit"])

    if page == "Make your prediction":
        one_sentence()
    elif page == "Examples":
        testing()
    elif page == "Example Streamlit":
        st.header("Encabezado")
        st.subheader("Subtitulo")
        st.write("Contenido")
        st.image("assets/uma.png")
    else:
        st.warning("An error has occured.")

if __name__=="__main__":
    main()