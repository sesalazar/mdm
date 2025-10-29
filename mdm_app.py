import streamlit as st
import numpy as np
import pandas as pd
import requests
import io
from openai import OpenAI
from scipy.spatial import distance
from ast import literal_eval

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
raw_url = "https://raw.githubusercontent.com/sesalazar/files/refs/heads/main/problemdefinitionembeddings_v2.csv"

@st.cache_resource
def get_client():
    return OpenAI()

@st.cache_data
def load_original_csv(url):                                                     #this gets csv file from my github
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))

def get_embedding(text):                                                        #this gets embeddings
    edited_text = text.replace("\n"," ")
    return client.embeddings.create(input=edited_text, model="text-embedding-3-small").data[0].embedding

st.set_page_config(page_title="CPT Problems for MDM")
df = load_original_csv(raw_url)

#building UI
st.write("### Problem Identifier")
st.write("##### Based on inputed text, output will be the most 'likely' problem.")
input_text = st.text_input("Input text to be analyzed:")
calc = st.button("Analyze")

#back end calculations
input_embedding = get_embedding(input_text)
matrix = np.array(df.DefinitionEmbeddings.apply(literal_eval).to_list())
x = []
if calc == True:
    for i in matrix:
        x.append(distance.euclidean(input_embedding,i))
    min_index = x.index(min(x))
    problem = df.at[min_index,"Problem"]
    st.write("Problem is: " + problem) 