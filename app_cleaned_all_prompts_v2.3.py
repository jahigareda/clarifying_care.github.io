# CHANGE HISTORY
#----------------------------------------------------------------------------
# V2.0 - Chris - Major UI change and source citation
# V2.1 - Jose / Shalini - Introduced Cuda and GPU processing to improve speed
# V2.2 - Chris - 1) Detect if Cuda is compatible and switch to CPU if not. 2) Changed chunk size from 1000 to 250 3) Lab results now format as a proper markdown table 4) Get a readability score at the bottom after the discharge is created.
#----------------------------------------------------------------------------

import pandas as pd
import torch
import textstat

import streamlit as st
st.set_page_config(layout="wide")

from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS   
from langchain.chains import RetrievalQA 
from langchain.schema.document import Document


llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.Q4_0.gguf",
    temperature=0.1,
    max_tokens=512,
    top_p=1,
    n_ctx=2048,
    n_threads=6
    #n_gpu_layers=20
    #callback_manager=callback_manager,
    #verbose=True,  # Verbose is required to pass to the callback manager
)

model_complete = False

# Function to calculate the readability scores of the input text
def show_readability_scores(original_text,model_text):
    st.markdown(calculate_readability_scores(original_text,model_text))
def calculate_readability_scores(original_text,model_text):
    """Function to calculate the desired readability scores when
    a text is passed to the function."""
    score_results = f'''
        |Readability Test|Original Discharge Summary|Clarifying Care|
        |---|---|---|
        |**dale_chall** | **{textstat.dale_chall_readability_score(original_text)}** | **{textstat.dale_chall_readability_score(model_text)}** |
        |flesch_reading_ease | {textstat.flesch_reading_ease(original_text)} | {textstat.flesch_reading_ease(model_text)} |
        |flesch_kincaid_grade | {textstat.flesch_kincaid_grade(original_text)} | {textstat.flesch_kincaid_grade(model_text)} |
        |gunning_fog | {textstat.gunning_fog(original_text)} | {textstat.gunning_fog(model_text)} |
        |SMOG_index | {textstat.smog_index(original_text)} | {textstat.smog_index(model_text)} |
        |consensus_score | {textstat.text_standard(original_text, float_output=False)} | {textstat.text_standard(model_text, float_output=False)} |
        |sentence_count | {textstat.sentence_count(original_text)} | {textstat.sentence_count(model_text)} |
        |word_count | {textstat.lexicon_count(original_text, removepunct=True)} | {textstat.lexicon_count(model_text, removepunct=True)} |
        |reading_time | {textstat.reading_time(original_text, ms_per_char=14.69)} | {textstat.reading_time(model_text, ms_per_char=14.69)} |
        '''
    return score_results
    
            

# Function to format source citation results
def format_source_citation(source_text):
    #Escape special characters that can cause issues with formatting
    ##source_text = source_text.replace("# ","\# ")
    #Split the source_documents result into each document
    ##source_list = list(source_text[1:-1].split("Document(page_content=")) 
    ##source_list.pop(0)
    source_list = source_text
    source_count = len(source_list)
    source_counter = 0
    formatted_text = ''
    #Loop through each document and create a string to show in the tool top
    for document in source_list:
        source_counter = source_counter+1
        source_without_linebreaks = str(document)
        source_without_linebreaks = source_without_linebreaks[14:-1].replace("# ","\# ")
        source_without_linebreaks = source_without_linebreaks.replace("\\n"," ")

        formatted_text = formatted_text + f'''
# Source {source_counter} of {source_count}
{source_without_linebreaks}

        '''

    return formatted_text
    

def intro():
    
    st.image('../Logo.png')
    #st.write("### Welcome to the Clarifying Care :sparkling_heart: Portal!")

    tab1, tab2 = st.tabs(["Demo", "Our Mission"])

    with tab1:
        def get_doctor_prompt(template,question):
            B_INST, E_INST = "<s>[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


            prompt_template =  B_INST + B_SYS + str(template) + E_SYS + str(question) + E_INST

            return prompt_template



        
        st.write(
            """
            Place in your own text and see the conversion to our simplified version following SBAR standards.
            """
        )


        notes = ""
        final_markdown = ""
        submitButton = False

        
        c1, c2= st.columns(2)
        with c1:
            with st.form(key='form1'):
                notes = st.text_area("Enter Hospital Notes :", height=350)
                submitButton = st.form_submit_button(label = 'Generate Summary')

        with c2:
            with st.container(border=True):
            
                chain_progress_bar = st.progress(0, text='Summary:')
                df_instructions = pd.read_excel(r"./Instructions_Template.xlsx","Instructions")
                


                if submitButton:
                    model_complete = False
                    #setup model stuff
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=100)
                    documents = [Document(page_content=x) for x in text_splitter.split_text(notes)]

                    all_splits = text_splitter.split_documents(documents)

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    model_name = "sentence-transformers/all-mpnet-base-v2"
                    if device.type == 'cuda':
                        model_kwargs = {"device": "cuda"}
                        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
                    else:
                        embeddings = HuggingFaceEmbeddings(model_name=model_name)


                    vectorstore = FAISS.from_documents(all_splits, embeddings)

                    chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectorstore.as_retriever(), return_source_documents=True)

                    final_markdown_list = []
                    final_sources_list = []
                    progress_counter = 0
                    progress_totalsteps = len(df_instructions)

                    for index, row in df_instructions.iterrows():

                        chain_progress_bar.progress(progress_counter/progress_totalsteps, text='Generating ' + str(row["Section"]) + ' Summary...' )
                        result = chain({"query": get_doctor_prompt(row["Prompt Template"],row["Question"])})

                        # Need to manually fix the Lab result to ensure it renders as a table. No amount of prompt engineering seems to get this right but I can get the headers to come out consistent so can manually add in the right table syntax.
                        if row["SBAR Item"] == "Labs":
                            result['result'] = result['result'].replace('| Date | Test Name | Result | Normal Range |',f'''Date | Test Name | Result | Normal Range |
                            --- | --- | --- | --- |''')

                        st.markdown(str(row["Pre-Markdown"]) + str(result['result']) + str(row["Post-Markdown"]), unsafe_allow_html=True, help=format_source_citation(result['source_documents']))
                        final_markdown = final_markdown + str(row["Pre-Markdown"]) + str(result['result']) + str(row["Post-Markdown"])
                        progress_counter = progress_counter + 1

                    model_complete = True
                    chain_progress_bar.empty()

                

                st.download_button(label="Download Summary",
                                data=final_markdown,
                                file_name="text.md",
                                mime='text/csv')
                
                st.markdown(f'''
                            
                            
                            #### Discharge Readability Score:
                            {calculate_readability_scores(notes,final_markdown)}

                            ''')
                    
                

        with tab2:    
            st.markdown(
                """
                Our mission is to simplify the hopsital discharge summary process by creating discharge summaries and making them more patient friendly.
                
                ### How do we accomplish our mission? 
                - Creating Discharge Summaries
                    - Prompt Engineering an LLM with hospital records, in this discharge notes in the dicharge csv from MIMIC IV Notes 
                - More Patient Friendly
                    - Following SBAR Format
                    - Validating the sumnmary requires a lower grade level of reading comprehention


                **Check the demo below** to see some examples
                of how we can simplify medical text!

                ### Text Demo: 
                - Place in your own text and see the conversion to our simplified version following SBAR standards

            """
            )

intro()