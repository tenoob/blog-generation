import streamlit as st
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import CTransformers

import time

#funtion to get responnse from llama
def get_response(input_text, word_len, blog_topic):
    start_time = time.time()

    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type = 'llama',
                        config = {'max_new_tokens': 256,
                                  'temperature': 0.02})
    
    template = """"
    write a blog for {blog_topic} job profic for a topic {input_text}
    within {word_len} words.
    """

    prompt = PromptTemplate(
        input_variables=["blog_topic","input_text","word_len"],
        template=template)
    
    #get response from llama
    response = llm(prompt.format(blog_topic=blog_topic, input_text = input_text, word_len=word_len))
    end_time = time.time()

    execution_time = end_time - start_time
    return response  , execution_time





st.set_page_config(page_title="Blog Generate",
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Blogs Generate")

input_text = st.text_input("Topic for Blog")

col1 ,col2 = st.columns([5,5])
with col1:
    words_len = st.text_input("Number of words")

with col2:
    blog_topic = st.selectbox('Blog topics',
                              ('Web dev','Data science','Others'),index=0)

submit = st.button("Generate")

if submit:
    response, exe_time = get_response(input_text,words_len,blog_topic)
    st.write(response)
    st.write(f"Exection_time: {exe_time} seconds")


