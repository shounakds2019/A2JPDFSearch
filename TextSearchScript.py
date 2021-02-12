# !pip install PyPDF2
# !pip install pyenchant
# !pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz --no-deps

import spacy
import PyPDF2
import os

# spacy english model (large)
nlp = spacy.load('en_core_web_lg')

from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.converter import TextConverter
import io
import sys
 
# total arguments
n = len(sys.argv)
#print("Total arguments passed:", n)
 
# Arguments passed
#print("\nName of Python script:", sys.argv[0])
 
# print("\nArguments passed:", end = " ")
# for i in range(1, n):
#     print(sys.argv[i], end = " ")

keyword = sys.argv[1]

# spacy english model (large)
nlp = spacy.load('en_core_web_lg')

from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.converter import TextConverter
import io

# method for reading a pdf file
def readPdfFile(filename, folder_name):
    
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open('./docs/'+filename, 'rb') as fh:

        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)

        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()


    return text

    # customer sentence segmenter for creating spacy document object
def setCustomBoundaries(doc):
    # traversing through tokens in document object
    for token in doc[:-1]:
        if token.text == ';':
            doc[token.i + 1].is_sent_start = True
        if token.text == ".":
            doc[token.i + 1].is_sent_start = False
    return doc

# create spacy document object from pdf text
def getSpacyDocument(pdf_text, nlp):
    main_doc = nlp(pdf_text)  # create spacy document object

    return main_doc

# adding setCusotmeBoundaries to the pipeline
nlp.add_pipe(setCustomBoundaries, before='parser')

from scipy import spatial

import enchant
d = enchant.Dict("en_US")

from spacy.tokens import Doc

# convert keywords to vector
def createKeywordsVectors(keyword, nlp):
    print("Searching document for word: ",keyword)
    doc = nlp(keyword)  # convert to document object
    #doc = Doc(nlp.vocab, words = keyword)  # convert to document object
    #print("doc",doc.vector)
    return doc.vector


# method to find cosine similarity
def cosineSimilarity(vect1, vect2):
    # return cosine distance
    return 1 - spatial.distance.cosine(vect1, vect2)


# method to find similar words
def getSimilarWords(keyword, nlp):
    similarity_list = []

    keyword_vector = createKeywordsVectors(keyword, nlp)

    for tokens in nlp.vocab:
        if (tokens.has_vector):
            if (tokens.is_lower):
                if (tokens.is_alpha):
                    similarity_list.append((tokens, cosineSimilarity(keyword_vector, tokens.vector)))

    similarity_list = sorted(similarity_list, key=lambda item: -item[1])
    similarity_list = similarity_list[:30]
    top_similar_words = [item[0].text for item in similarity_list]
    top_similar_words = top_similar_words[:3]
    top_similar_words.append(keyword)
    #print("top_similar_words:1",top_similar_words)
#     for token in Doc(nlp.vocab, words = keyword):
#         top_similar_words.insert(0, token.lemma_)

#     for words in top_similar_words:
#         if words.endswith("s"):
#             top_similar_words.append(words[0:len(words)-1])

#     top_similar_words = list(set(top_similar_words))
    #print("top_similar_words:",top_similar_words)
    #top_similar_words = [words for words in top_similar_words if enchant_dict.check(words) == True]
    top_similar_words = [words for words in top_similar_words if d.check(words) == True]
    
    top_similar_words = [words for words in top_similar_words]
    similarity_list.clear()
    return ", ".join(top_similar_words)


import gensim,nltk
from nltk.data import find
nltk.download("word2vec_sample")
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

#keywords = ["safety","guidelines","manufacturers"]
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
similar_keywords_list = []
similar_keywords = ""
#keyword = "justice"
#similar_keywords = getSimilarWords(keyword, nlp)
for word_tuple in word2vec_model.similar_by_word(keyword)[:5]:
    similar_keywords_list.append(word_tuple[0])
    
similar_keywords = " ".join(similar_keywords_list)
print("Similar words are:",similar_keywords)

from spacy.matcher import PhraseMatcher
from scipy import spatial

# method for searching keyword from the text
def search_for_keyword(keyword, doc_obj, nlp):
    phrase_matcher = PhraseMatcher(nlp.vocab)
    li = list(keyword.split(" ")) 
    print(li)
    patterns = [nlp.make_doc(text) for text in li]
    phrase_matcher.add("TerminologyList", patterns)

    matched_items = phrase_matcher(doc_obj)
    print(matched_items)
    matched_text = []
    for match_id, start, end in matched_items:
        text = nlp.vocab.strings[match_id]
        span = doc_obj[start: end]
        matched_text.append(span.sent.text)
    
    for txt in matched_text:
        print("Found Match ####")
        print(txt)
    

pdf = readPdfFile("doc.pdf","docs")
doc_obj = getSpacyDocument(pdf,nlp)
print("Searching for...",similar_keywords,"\n")
search_for_keyword(similar_keywords,doc_obj, nlp)
     
