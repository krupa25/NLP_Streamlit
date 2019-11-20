import streamlit as st

#NLP Packages
import nltk
nltk.download('punkt')
import spacy
from textblob import TextBlob
from gensim.summarization import summarize 
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from gensim.summarization import keywords




def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,3)
    summary_list = [str(sentence) for sentence in summary]
    result = ''.join(summary_list)
    return result


def text_analyzer(my_text):
    nlp = spacy.load('en')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    allData = [('"Tokens":{},"Lemma":{}'.format(token.text,token.lemma_)) for token in docx]
    return allData

def entity_analyzer(my_text):
    nlp = spacy.load('en')
    docx = nlp(my_text)
    entities = [(entity.text,entity.label_) for entity in docx.ents]
    return entities

def main():
    #NLP App with Streamlit
    st.title("NLPiffy your Text")
    st.subheader("Natural Language Processing on the Go")

    #Tokenization
    if st.checkbox("Tokens and Lemma"):
        st.subheader("Tokenize the text")
        text = st.text_area("Enter your text:","Type here...")
        if st.button("Analyze"):
            nlp_result = text_analyzer(text)
            st.json(nlp_result)




    #Named Entity
    if st.checkbox("Named Entities"):
        st.subheader("Extract the named entities")
        text = st.text_area("Extract the entities","Type here...")
        if st.button("Extract"):
            nlp_result = entity_analyzer(text)
            st.json(nlp_result)
    #Sentiment Analysis
    if st.checkbox("Sentiment Analysis"):
        st.subheader("Sentiment Analysis of your Text")
        text = st.text_area("Enter your text","Type here...")
        if st.button("Analyze"):
            blob = TextBlob(text)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)


    #Text Summarization
    if st.checkbox("Text Summarizer"):
        st.subheader("Summarize your Text")
        text = st.text_area("Enter your text","Type here...")
        summary_options = st.selectbox("Select your Summarizer",("gensim","sumy"))

        if st.button("Summarize"):
            if summary_options=="gensim":
                st.text("Using Gensim...")
                summary_result = summarize(text)
            elif summary_options=="sumy":
                st.text("Using sumy...")
                summary_result = sumy_summarizer(text)

            else:
                st.warning("Using Default Summarizer")
                st.text("Using Gensim...")
                summary_result = summarize(text)

            st.success(summary_result)

    #KEYWORDS
    if st.checkbox("Keywords Generator"):
        st.subheader("Generate Keywords from your text")
        text = st.text_area("Enter your text","Type here...")
        
        if st.button("Generate"):
            keyw = keywords(text=text).split('\n')
            st.success(keyw)


    st.sidebar.subheader("About the App")
    st.sidebar.text("You can apply various NLP techniques on your text.")





if __name__=='__main__':
    main()