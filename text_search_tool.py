from db_helpers import (select_from_db_table, 
                        insert_into_db_table,
                        create_sentences_table) 
from text_embedding_tool import BertTfidfEmbeddingTool
import nltk
import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import string
from tqdm import tqdm


class TextSearchTool(BertTfidfEmbeddingTool):

    def __init__(self, bert_model_dir, do_lower_case, max_seq_length=128, 
                layer_indexes=[-1, -2, -3, -4], verbose=0):
        
        super(TextSearchTool, self).__init__(
            bert_model_dir, do_lower_case, max_seq_length, 
            layer_indexes, verbose)
        
        self.sentences = pd.DataFrame([])
        self.embeddings = np.array([])

  
    def _split_text_to_sentences(self, texts, text_ids):

        # download nltk library to extract sentences from paragraphs
        nltk.download('punkt')
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        sentences_df = {
            'sentence_no': [],
            'text_id': [],
            'sentence': []
        }
        for text_id, text in zip(text_ids, texts):
            sentences = sentence_tokenizer.tokenize(text)
            if self.do_lower_case:
                sentences = [sentence.lower() for sentence in sentences]
            sentences_df['sentence'] += sentences
            sentences_df['sentence_no'] += list(range(len(sentences)))
            sentences_df['text_id'] += [text_id] * len(sentences)
    
        return pd.DataFrame(sentences_df)


    def fit(self, texts, text_ids):

        sentences = self._split_text_to_sentences(texts=texts, 
                                                  text_ids=text_ids)
        embeddings = self.transform(sentences.sentence.values)
        self.sentences, self.embeddings = sentences, embeddings


    def upload_to_db(self, texts, text_ids, db_filepath, batch_size=10000):

        sentences_df = self._split_text_to_sentences(texts=texts,
                                                     text_ids=text_ids)

        # check if db already exists
        if os.path.isfile(db_filepath):
        # if so check the existing entries' keys to prevent duplicates
            db_sentence_keys = select_from_db_table(
                columns=['text_id','sentence_no'], 
                table='sentences', 
                db_filepath=db_filepath
            )
            completed_keys_mask = sentences_df.apply(
                lambda x: (x.text_id, x.sentence_no) in db_sentence_keys,
                axis=1 
            )
            percent_already_loaded = np.mean(completed_keys_mask) * 100
            print(f"{percent_already_loaded:.0f}% of data already loaded")
            sentences_to_load = sentences_df[~completed_keys_mask]
            if sentences_to_load.empty:
                print("All sentences already uploaded to database")
                return None
        else:
            # otherwise create a new database to store data
            create_sentences_table(db_filepath)
            sentences_to_load = sentences_df
        
        if not self.vectorizer:
            print("Vectorizer not fitted. Fitting vectorizer from data Provided.")
            print("WARNING: This may not align with the vectorizer used to fit"
                  + " existing entries in the database")
            self.fit_tfidf_vectorizer(sentences_df)
        
        # separate the sentences to be loaded into batches
        total_batches = len(sentences_to_load) // batch_size + 1
        for i in range(total_batches):
            batch_sentences = sentences_to_load[i*batch_size:(i+1)*batch_size]

            # calculate bert/tfidf vector for each example
            batch_embeddings = self.transform(batch_sentences.sentence.values)

            # create tuples for each entry to be uploaded to database
            batch_db_entry_tuples = []
            for idx, sentence_entry in enumerate(batch_sentences.itertuples()):
                entry_tuple = (sentence_entry.sentence_no, 
                            sentence_entry.text_id,
                            sentence_entry.sentence,
                            *batch_embeddings[idx])
                batch_db_entry_tuples.append(entry_tuple)

            # load to db
            insert_into_db_table(values=batch_db_entry_tuples, 
                                 table='sentences', 
                                 db_filepath=db_filepath)
        return None

    def fit_from_db(self, db_filepath):
        db_output = select_from_db_table(columns=["*"], 
                                         table='sentences', 
                                         db_filepath=db_filepath)
        sentences = pd.DataFrame(
            list(map(lambda x: x[:3], db_output)),
            columns = ['sentence_no', 'text_id', 'sentence']
        )
        embeddings = np.array(list(
            map(lambda x: x[3:], db_output)
        ))
        self.sentences, self.embeddings = sentences, embeddings
        self.fit_tfidf_vectorizer(sentences.sentence.values)

    def _check_cosine_similarity(
        self, query_embedding, comparison_embeddings, top_n):

        # split into batches to avoid calcluating a massive n x n cosine matrix
        BATCH_SIZE = 200
        query_embedding = query_embedding.reshape(1,-1)
        query_cosine_similarity = np.array([])
        epochs = len(comparison_embeddings) // BATCH_SIZE + 1
        for i in tqdm(range(epochs)):
            batch_comparison_embeddings = comparison_embeddings[i*BATCH_SIZE:
                                                                (i+1)*BATCH_SIZE]
            # create a cosine similarity matrix with query embedding at position zero
            batch_embedding_matrix = np.concatenate(
                [query_embedding, batch_comparison_embeddings]
            )
            batch_cosine_similarity = (
                cosine_similarity(batch_embedding_matrix)[1:,0])
            # concatenate the similarity values for the comparison embeddings to the output
            query_cosine_similarity = np.concatenate(
                [query_cosine_similarity, batch_cosine_similarity]
            )
        # return a ranked vector of top_n most similar embeddings by index
        most_similar = list(reversed(
            np.argsort(query_cosine_similarity)
        ))
        return most_similar[:top_n]


    def query(self, query_text, top_n=10):

        if self.sentences.empty or not self.embeddings.any():
            raise Warning("No text data to query against. Call fit and try again")
        elif not self.vectorizer:
            raise Warning("""
                TF-IDF vectorizer has not been fitted.
                Query reqires TF-IDF vectorizer to calculate similarity
                """)
        
        query_embedding = self.transform([query_text])
        top_n_most_similar_idxs = self._check_cosine_similarity(
            query_embedding=query_embedding, 
            comparison_embeddings=self.embeddings, 
            top_n=top_n)
        most_similar = self.sentences.iloc[top_n_most_similar_idxs]
        print(top_n_most_similar_idxs)

        print("=" * 100)
        print(f"Query '{query_text}'")
        print("Is most similar to:\n")
        for idx, entry in enumerate(most_similar.itertuples()):
            print(f"{idx+1}. {entry.sentence}")
            print("=" * 100)
        return most_similar