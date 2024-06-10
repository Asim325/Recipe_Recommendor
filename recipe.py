import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## Function To get response from LLAma 2 model

def getLLamaresponse(weather, input_text, no_words, name, style):

    ### LLama2 model
    llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q2_K.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})
    
    ## Prompt Template
    template = """
        Write a recipe for {weather} chef's for ingredients {input_text}
        within {no_words} words and cuisine style {style} generate recipe {name} with specific name.
            """
    
    prompt = PromptTemplate(input_variables=["weather", "input_text", "no_words", "name", "style"],
                            template=template)
    
    ## Generate the response from the LLama 2 model
    response = llm(prompt.format(weather=weather, input_text=input_text, no_words=no_words, name=name, style=style))
    print(response)
    return response


st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Recipe Blogs 🤖")

input_text = st.text_input("Enter the ingredients")

## creating to more columns for additional 2 fields

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
    name = st.text_input('Recipe Name')
with col2:
    weather = st.selectbox('Writing the Recipe for weather type',
                            ('Hot', 'Cool', 'Moderate'), index=0)
    style = st.selectbox('Cuisine Style',
                            ('Italian', 'Mexican', 'Indian', 'Pakistani'), index=0)
    
submit = st.button("Generate Recipe")

## Final response
if submit:
    st.write(getLLamaresponse(weather, input_text, no_words, name, style))
