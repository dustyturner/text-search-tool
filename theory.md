# Bert / TF-IDF Document Search Tool

There's a lot out there to be read. Every day, we are committing more words to (virtual) pages than have been recorded over the lifetimes of folk that just had books to contend with. The more we write, the more we contribute to a big problem - how do we find the information we need in all this text?

This is one of the fundamental issues that Natural Language Processing (NLP) attempts to address. If we can distil text into a form that can be understood by computers (and the meaning of "understanding" here is a complex one), we can wade through the heaps and heaps of noise to find a signal. Well, the computers can do it for us while we drink tea.

This document search tool was envisaged as a response to the issues faced by researchers during the Covid-19 outbreak. Huge volumes of academic work were being published with scarce little time available for researchers to digest information that may help them in their own research. The tool can take blocks of text and search for relevant information from a user-defined query.

## Basic Theory

The theoretical components of the search process adopted by this tool will be familiar to initiates of NLP. We start with the bedrock of pretty much all modern NLP - embeddings: 

### embeddings

An embedding is a numerical representation of a qualitative variable. In the context of NLP the variable will be a word or collection of words, but this can just as easily be a categorical value (like a color) or a picture, or a voice recording. The key feature of an embedding is that it captures the intrinsic information contained in a qualitative input and represents it numerically in a uniform output - a fixed length vector.

As simple example, take the embedding of the words **King, Prince, Queen, Princess** as follows:

<div style="text-align:center"><code>King = [-1,1]</code></div>
<div style="text-align:center"><code>Prince = [-1,-1]</code></div>
<div style="text-align:center"><code>Queen = [1,1]</code></div>
<div style="text-align:center"><code>Princess = [1,-1]</code></div>

Each word is represented by a two value vectorized encoding. In this example, the first dimension can be seen to encode the **masculinity / femininity** of the word, and the second dimension the seniority relationship **junior / senior** - the lower / higher values denoting the position of each word on these conceptual scales.

In practice, embeddings do not usually demonstrate these clear relationships between the numeric values and qualitative concepts. The values are calculated by an algorithmic process aiming to quantify the commonalities of each word or item within your qualitative variable - this is often in the context of a supervised learning task, where a model will attempt to predict an output given a particular input. Items closely associated with one another in the context of the learning task (i.e. inputs closely associated with certain target outputs) will share similar numerical values in their embeddings, those that share no association will show much greater differences. 

Armed with embeddings for our qualitative data, we can then make quantitative numerical comparisons between examples where this was not possible before. This leads us to our next key concept:

### vector comparison

Once we have a set of numerical embedding vectors, we can compare them using a number of techniques familar to anybody that has learned any co-ordinate geometry or multivariate calculus. The two most commonly used are euclidean distance and cosine similarity:

* Euclidean distance:
    This is the length of the line that can be drawn between the two co-ordinate points of our vectors. The more similar the values in each vector, the closer they will be together and the lower their euclidean distance will be. Simple!

* Cosine Similarity:
    The cosine similarity is the magnitude of the angle between our two vectors - effectively measuring the similarity of the direction they both point in. Much like with euclidean distance, the cosine similarity will be smaller the more similar the values in each vector are. Unlike euclidean distance, cosine similarity ignores the magnitude of each vector, and so both vectors can have very different values, and still have a very low cosine similarity provided they point in similar directions.

The best choice depends on the particular use-case. Euclidean distance is very useful when the distribution of your variables is uniform or relatively normal in distribution - for example, comparing the similarity of rgb color values. Basically any apples-to-apples type comparison will usually be best calculated by euclidean distance.

Cosine similarity is often better in situations where the magnitude of your vectors is less important. A perfect example is the context of this discussion - text (or multiple word) comparison. Often, we will see two statements with a very similar meaning, but with significantly different lengths i.e. one statement uses many more words than the other. Such statements may produce embeddings with very different magnitudes - the sentence with more words has much higher co-ordinate values than the other, as the additional words compound the embedding values. The individual words that define each statement, however, may be very similar and so the vectors point in very similar directions. Cosine similarity makes allowances for this phenomenon.

It shouldn't come as a surprise that the cosine similarity will be used for this search method. But we have yet to establish how we will calculate our embeddings, without which we're not going to get very far. This leads us to the two embedding techniques that will be used in this search tool:

* ### Bert:

    Bert is a hugely powerful and versatile Natural Language Processing model. It has been trained on a gigantic corpus of text from Wikipedia and BooksCorpus, and has achieved state of the art results on a wide range of NLP benchmarks with little task-specific training. Not only can Bert accurately capture the meaning of individual words as they appear in a sentence, but also their meaning in the context of the other words around them i.e. the distinction between "I am happy not sad" and "I am sad not happy" - historically this has been a big pitfall for other les sophisticated models. 
    
    Unlike the majority of NLP solutions that came before it, bert shuns the dominant recurrent architectures in favour of a "bidirectional transformer network". Whereas recurrent neural networks (RNNs) (i.e. LSTMs and the like) encode a sequence one token at a time left -> right, bert uses a transformer network with "self-attention" layers to "attend" to the whole sequence at once, doing so each time it makes an output prediction.
    
    In an attempt to paraphrase at a high level: 
    
    RNN models encode and input one -> word -> at -> a -> time, in a path moving from one word to the next, and then decode this input one -> word -> at -> a -> time to make a prediction. When making inferences, if the prediction requires the model to look at the beginning of the input sentence, the model will have to look back along this path from output to the relevant input to find the information it needs. This is a long way for a signal to travel, meaning such networks struggle to "remember" certain information, like the context provided by words that are at a distance from one another in a sentence. Take, for example, the color of the dog in the sentence "*Peppa, my dog* and best friend in the whole wide world, is a *brown* terrier" - the signal has to travel a long way between the words "dog" and "brown" to be interpreted by the decoder.

    Bert avoids this issue by making use of Transformer modules. Transformers use a technique called the attention mechanism to allow a model to "focus" on a specific part of a sentence as required. Rather than inputting words one-by-one, the transformer model accepts the simultaneous input of every word along with an embedding that encodes their respective positions. The model then trains an "attention layer" that uses a set of key/value weightings that allow the model to "attend" to individual words from the input directly, regardless of the distance between the words and without the signal having to travel through all of the following words to be decoded. 

* ### TF-IDF vectors

    If Bert is a top-of-the-line PC (or mac, whatever floats your boat), then Term Frequency / Inverse Document Frequency (or TF-IDF) is a pocket calculator - still an important and useful bit of kit, but many orders of magnitude less complex. Instead of encoding a complex graph of calculations like a neural network, TF-IDF very simple mathematical formula that uses the counts of the appearances of each word (or word combinations) in a group of documents to "score" their importance to each individual document. 

    The Term Frequency is the number of times a query word appears in a document, divided by the number of words in the document. So the term frequency of "this" in:

    "This is an example of the term frequency of this document"

    is 2 (appearances of "this") / 11 (total words), simple! The Term Frequency can be thought of as capturing the importance of a word in a document - the more times it appears the more important it is and the higher score it gets.

    Inverse Document Frequency is the number of documents, divided by the number of documents containing a given query word. So the inverse document frequency of "this" in the following group of sentences is:

    "This is another example, this time demonstrating IDF"

    "This is hopefully making sense"

    "I think you're getting the idea"

    Is 3 (total documents) / 2 (documents containing "this"). The Inverse Document Frequency can be though of as controlling for generally common words (i.e. "the", "is", "and", "this") that don't convey a lot of meaning, preventing them from dominating our estimates of the importance of different words. 

    So the overall TF-IDF for the word "this" in the first sentence of our document is (2 / 11) * (3 / 2). In the wild, TF-IDF values are augmented with smoothing and normalisation techniques to make them more effective, so you are unlikely to see them calculated like this, but you get the idea.

    By calculating a TF/IDF score for every unique word that appears over whole corpus (group of documents) once for each document, we get a vectorized representation of the document's contents. This is referred to as a "sparse-vector" representation, as many of the words in the vocabulary (if it is large) will not appear in a given document, and so the vectors will contain a large proportion of zero values. These TF-IDF vectors can be used in much the same way as the Bert vector embeddings to compare the similarity of different texts.

## Search Tool Recipe

We can combine these two techniques for encoding written information to create a powerful search tool. The following is a step-by-step recipe for this approach:

1. ### Split texts into sentences:
    The principal "unit" over which the tool will be searching is individual sentences from each document. As such, the first step is to split each document into a list of sentences which can be identified from the document in which they originate and their position in that document.

2. ### Split sentences into tokens:
    The next step is lifted straight from the bert methodology. Like all deep-learning approaches to NLP, the bert model has a "vocabulary" that allows it to encode every input into a sequence of numerical "tokens", over which it can then perform lots of calculations. A large portion of this vocabulary is composed of common words we would all be very familiar with. Other longer, more complex, or just less common words are sub divided into common syllables. Those words or syllables the model doesn't recognise are then broken into individual characters.

    This can, perhaps, be most easily seen from an example. Bert might break down the sentence:

    "my dog, peppa, is actually a staffordshire bull terrier"

    into the following:

    ['my', 'dog', ',', 'pep', '##pa', 'is', 'actually', 'a', 'staff', '##ords', '##hire', 'bull', 'terr', '##ier']

    As you can see, this "tokenization" process breaks up the less common words into syllables - it also prefixes the syllables with "##" so they can be distinguished from individual words i.e. the words / syllables "on" and "a" in "on a bench" -> ["on", "a", "bench"] and "Barcalona" -> ["Barc" "##al", "##on", "##a"]. These tokens can then be numerically encoded and fed to Bert.

3. ### Calculate bert output embeddings for tokens:
    Once a sentence has been tokenized and encoded, it is fed to the Bert model which calculates a number of output activations from the transformer layers of the Bert model. Bert would ordinarily use these activations to make predictions for a given task, but in this case we use the activations themselves as a numerical representation of each token in the sentence. The authors of the Bert paper recommend using an average of the last four layers outputs, and so this is what is used to create an output embedding for each token. 


4. ### Calculate TF-IDF values for tokens:
    Along with our Bert output embeddings, we can also use the TF-IDF procedure to calculate a score for each of the tokens in each sentence. This should be fairly self explanatory based on the process described in the theory section above - the only difference being that a score is calculated for each token, which might be a sub-section of a word, rather than the words as a whole.

5. ### Combine Bert token embeddings and TF-IDF token scores to form sentence-level embeddings:
    With the token-level bert embeddings and the TF-IDF values in hand, we can now combine these values to form a weighted embedding for each sentence. Basically, multiply each token embedding by it's respective TF-IDF value and then average over the token embedding to produce a single sentence-level embedding.

6. ### Calculate a query embedding using the same process:
    Once we have an embedding for each sentence, we can then use this information to locate the most similar sentence to a given text query.  To do so, we first take the query sentence in question and calculate it's embedded representation using steps 2-5 above. 

    Note: This query will need to be phrased as keyword search (i.e. "most popular pet name") rather than as a question (i.e. "what is the most popular pet name?"). Bert may otherwise identify the most similar sentences as those phrased as questions, rather than the sentences with the most similar content that we are looking for.

7. ### Compare cosine similarity of the query and document sentences:
    With an embedded query, we can then compare the numerical properties of the embedded sentences to find the most similar sentences. We calculate the Cosine similarities of each of the embedded sentences and return the sentences with the highest similarities as search results.

    There are a number of potential techniques to make this comparison - when comparing vectorized embeddings, the most common are euclidean distance and cosine similarity. We use cosine similarity here as, in the context of sentence embeddings, the euclidean distance is biased toward sentences that use the same words irrespective of meaning, whereas the cosine similarity better captures sentences that use different words but have a similar meaning. 

    It would take a while to explain the mathematics behind these principles but a high level intuition can be built relatively quickly. The cosine similarity compares the direction that two vectors are facing in as they stretch out into multi-dimensional space. Euclidean distance, on the other hand, measures the distance between the vector co-ordinates. Sentences that use similar words are more likely to fall closer to one another in vector space, but sentences using similar words may have very different meanings 


### advantages

### disadvantages