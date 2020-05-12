from bert.modeling import BertConfig
from bert import tokenization
from bert.extract_features import (input_fn_builder, 
                                   InputExample, 
                                   model_fn_builder,
                                   InputFeatures, 
                                   convert_examples_to_features)
import numpy as np
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.ERROR)

class BertTfidfEmbeddingTool(object):
  
    def __init__(self, bert_model_dir, do_lower_case, max_seq_length=128, 
               layer_indexes=[-1, -2, -3, -4], verbose=0):
        self.bert_model_dir = bert_model_dir
        self.do_lower_case = do_lower_case,
        self.max_seq_length = max_seq_length
        self.layer_indexes = layer_indexes
        self.verbose = verbose
        self.vectorizer = None

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=bert_model_dir + 'vocab.txt', 
            do_lower_case=do_lower_case)
  
    def transform(self, input_sentences):

        estimator = self._build_bert_model()
    
        input_fn, input_features_dict = self._build_bert_input(input_sentences)

        tfidf_weights_dict = self._build_tfidf_weights_dict(input_sentences,
                                                            input_features_dict)
        output = []
        for result in tqdm(estimator.predict(input_fn, 
                                                yield_single_examples=True),
                        total = len(input_sentences)):
            # each result comes in dict with inputexample id as primary key
            # each example has sub dict with layer outputs as keys, 
            # shape (max_seq_length, hidden_layer_size = 768 in this case)

            # model predict output is effectively shape: 
            # (n_examples, n_layers, max_seq_length, hidden_layer_size)
            
            # colects unique id from estimator output and input feature from dict
            example_id = int(result["unique_id"])
            input_feature = input_features_dict[example_id]
            example_tfidf_weights = tfidf_weights_dict[example_id]
            
            # averages output over all layer indexes
            mean_layers_output = np.mean(
                [result["layer_output_%d" % n] for n in range(len(self.layer_indexes))],
                axis=0
            )
            example_token_outputs = []
            # loops through tokens in example sentence
            for (i, token) in enumerate(input_feature.tokens):
                # collects output vector for specific token in mean_layer_output
                values = [
                    round(float(x), 6) for x in mean_layers_output[i:(i + 1)].flat
                ]
                # weight token's embedding output by tfidf value for token
                values = np.array(values) * example_tfidf_weights[i]
                # add output to list of all weighted token embeddings
                example_token_outputs.append(values)
                # calculate mean embedding for full sentence
            example_sentence_output = np.mean(example_token_outputs, axis=0)
            output.append(example_sentence_output)
        return np.array(output)

  
    def fit_tfidf_vectorizer(self, sentences):
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer.tokenize, 
                                          norm=None)
        self.vectorizer.fit(sentences)


    def _build_bert_model(self):
        # load pre-trained model config
        bert_config_file = self.bert_model_dir + "bert_config.json"
        bert_config = BertConfig.from_json_file(bert_config_file)

        # code to facilitate TPU usage - not used in this case so can be overlooked
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            master=None,
            tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=8,
            per_host_input_for_training=is_per_host)
        )

        # then load build BERT model
        checkpoint_file = self.bert_model_dir + 'bert_model.ckpt'
        
        model_fn = model_fn_builder(
            bert_config = bert_config,
            # the bert_model.ckpt file is actually three files, but is referenced as one
            init_checkpoint = checkpoint_file,
            layer_indexes = self.layer_indexes,
            use_tpu = False,
            # extract_features script reccomends this to be set to true if using TPU
            # apparently much faster
            use_one_hot_embeddings = False
        )
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            predict_batch_size=32
        )

        return estimator


    def _build_bert_input(self, input_sentences):
        # first convert raw text sentences into a list of InputExample objects
        input_examples = []
        # in this case use single sentence InputExamples for single embedding
        # loop through each full example from input df
        for idx, sentence in enumerate(input_sentences):
            input_example = InputExample(
                unique_id = idx,
                text_a = sentence,
                text_b = None
            )
            input_examples.append(input_example)

        # then convert the InputExample objects to InputFeature objects
        input_features = convert_examples_to_features(
            examples=input_examples, 
            seq_length=self.max_seq_length, 
            tokenizer=self.tokenizer
        )
        # build input_fn to feed to bert model
        input_fn = input_fn_builder(
            features=input_features, 
            seq_length=self.max_seq_length
        )
        # store required info from each example for use in predict loop
        input_features_dict = {}
        for feature in input_features:
            input_features_dict[feature.unique_id] = feature
        
        return input_fn, input_features_dict
  

    def _build_tfidf_weights_dict(self, input_sentences, input_features_dict):

        if not self.vectorizer:
            print("\n" + "=" * 100)
            print("No fitted vectorizer provided.")
            print("Fitting TF-IDF vectorizer to input sentences")
            print("=" * 100)

        self.fit_tfidf_vectorizer(input_sentences)

        tfidf_feature_names = self.vectorizer.get_feature_names()
        tfidf_matrix = self.vectorizer.transform(input_sentences).toarray()
        tfidf_weights_dict = {}

        for unique_id, feature in input_features_dict.items():
            tfidf_feature_weights = []
            tfidf_feature_vector = tfidf_matrix[unique_id]
            for token in feature.tokens:
                if token in ["[CLS]", "[SEP]"]:
                    tfidf_feature_weights.append(0)
                    continue
                try:
                    token_idx = tfidf_feature_names.index(token)
                except:
                    token_idx = None
                # append the token's tfidf value to the vector for this sentence
                if token_idx:
                    tfidf_feature_weights.append(tfidf_feature_vector[token_idx])
                else:
                    tfidf_feature_weights.append(1)
            tfidf_weights_dict[unique_id] = tfidf_feature_weights
        
        return tfidf_weights_dict
