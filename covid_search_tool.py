import nltk
import pandas as pd
import string
from text_search_tool import TextSearchTool

class CovidSearchTool(TextSearchTool):

    # parent split method takes two arguments: text_id and text
    # create a blank second argument _ so that parent upload_to_db works 
    def _clean_covid_abstracts(self, abstracts_df):
        # remove duplicate cord_uid papers:
        abstracts_df.drop_duplicates(subset='cord_uid', inplace=True)

        # remove papers with placeholders for abstract
        abstracts_df.dropna(subset=['abstract'], inplace=True)
        no_abstracts = abstracts_df.abstract.isin(['Unknown', 'Abstract No Abstract'])
        abstracts_df = abstracts_df[~no_abstracts]

        # remove duplicate papers based on title and authors
        # function to remove caps and any punctuation
        def convert_text(text):
            converted = (text.lower()
                        .translate(str.maketrans('','',string.punctuation)))
            return converted
        
        # remove caps / punctuation from title and authors
        for col in ['title', 'authors']:
            abstracts_df[col] = abstracts_df[col].apply(
                lambda x: convert_text(x) if type(x) == str else x
            )
        
        abstracts_df.drop_duplicates(subset=['title','authors'], inplace=True)

        # isolate texts (abstracts) and text_ids (cord_uids) 
        text_ids = abstracts_df.cord_uid.values
        texts = []
        # prepend paper title to texts
        for entry in abstracts_df.itertuples():
            text = entry.title
            if text[-1] != ".":
                text += "."
            text += " "
            text += entry.abstract
            texts.append(text)

        return texts, text_ids

    def fit(self, abstracts_df):
        texts, text_ids = self._clean_covid_abstracts(abstracts_df)
        super().fit(texts=texts, text_ids=text_ids)

    def upload_to_db(self, abstracts_df, db_filepath, batch_size=10000):
        texts, text_ids = self._clean_covid_abstracts(abstracts_df)
        super().upload_to_db(texts=texts, 
                             text_ids=text_ids,
                             db_filepath=db_filepath,
                             batch_size=batch_size)