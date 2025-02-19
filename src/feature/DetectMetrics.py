import re
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from nltk.corpus import stopwords
import textstat
import spacy
from textblob import TextBlob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter
import math

class DetectMetrics:
    def __init__(self, text, nlp, CLEAN=1):

        cleaned_text = re.sub(r'[\r\n]+', ' ', text)  


        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        self.text = cleaned_text
        self.words =word_tokenize(self.text)

        
        self.stop_words = set(stopwords.words('english'))
        self.nlp = nlp
        self.doc = self.nlp(self.text)
        self.sentences = [sent.text for sent in self.doc.sents]
        self.word_count = len(self.words)
        self.sent_count = len(self.sentences)


    
    
    def Average_Word_Length(self):
        calculated_words = [word for word in self.words if word.isalpha()]
        if len(calculated_words) == 0:
            return 0
        total_length = sum(len(word) for word in calculated_words)
        return total_length / len(calculated_words)
    
    def Average_Sentence_Length(self):
        if len(self.sentences) == 0:
            return 0
        calculated_words_sent = []
        for sentence in self.sentences:
            # calculated_words = [word for word in word_tokenize(sentence) if word.isalpha()]
            calculated_words = [word for word in word_tokenize(sentence)]
            calculated_words_sent.append(calculated_words)
        sentences_len = [len(i) for i in calculated_words_sent]
        # total_words_in_sentences = sum(len(re.findall(r'\w+', sentence)) for sentence in self.sentences)
        return sum(sentences_len) / len(self.sentences)

    def Stopword_Ratio(self):
        stopwords_in_text = [word for word in self.words if word.lower() in self.stop_words]
        stopword_count = len(stopwords_in_text)
        return stopword_count / self.word_count if self.word_count > 0 else 0
    
    def Long_Sentence_Ratio(self, min_length=20):
        if len(self.sentences) == 0:
            return 0
        long_sent = 0
        for sent in self.sentences:
            if len(word_tokenize(sent))>min_length:
                long_sent = long_sent + 1

        return long_sent / len(self.sentences)
    
    def Long_Word_Ratio(self, min_length=10):
        long_words = [word for word in self.words if len(word) >= min_length and word.isalpha()]
        long_word_count = len(long_words)
        return long_word_count / self.word_count if self.word_count > 0 else 0
    
    def Readability_Sore(self):
        return textstat.flesch_reading_ease(self.text)

    def Type_Token_Ratio(self):
        tokens = [token.text.lower() for token in self.doc if token.is_alpha]  # 只考虑字母
        tokens = [i for i in tokens if i.isalpha()]
        types = set(tokens) 
        return len(types) / len(tokens) if len(tokens) > 0 else 0
    
    def Dependency_Relation_Variety(self):


        dep_relations = [token.dep_ for token in self.doc if token.dep_ != "punct"]  # 排除标点符号

        dep_relation_count = Counter(dep_relations)
        variety = len(dep_relation_count)

        entropy = 0
        total_tokens = len(dep_relations)
        for count in dep_relation_count.values():
            prob = count / total_tokens
            entropy -= prob * math.log2(prob)
        
        return variety
    
    def Sentiment_Polarity(self):
        if len(self.sentences)==0:
            return {"polarity": 0, "subjectivity": 0}
        total_polarity = 0
        total_subjectivity = 0
        for sent in self.sentences:
            blob = TextBlob(sent)
            total_polarity += blob.sentiment.polarity
            total_subjectivity += blob.sentiment.subjectivity

        return {"polarity": total_polarity/len(self.sentences), "subjectivity": total_subjectivity/len(self.sentences)}

    def Subordinate_Clause_Density(self):
        if len(self.sentences) == 0:
            return 0
        total_subordinate_clauses = 0

        for sentence in self.doc.sents:
            for token in sentence:
                if token.dep_ in ["advcl", "ccomp", "xcomp", "relcl", "acl"]:  # 常见的从句依存关系  # 这个不太对
                    total_subordinate_clauses += 1

        return total_subordinate_clauses / len(self.sentences)

    
    def get_metrics(self):
        return {
            "AWL": self.Average_Word_Length(),
            "ASL": self.Average_Sentence_Length(),
            "SWR": self.Stopword_Ratio(),
            "LWR": self.Long_Word_Ratio(),
            'FRE': self.Readability_Sore(),
            'TTR': self.Type_Token_Ratio(),
            'DRV': self.Dependency_Relation_Variety(),
            'PS': self.Sentiment_Polarity()['polarity'],
            'SS': self.Sentiment_Polarity()['subjectivity'],
            'SCD': self.Subordinate_Clause_Density(),
            'LSR': self.Long_Sentence_Ratio()
        }

class Review_SEG:
    def __init__(self, text, nlp):
        self.text = text
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s\.,!?;:()-]', '', text) # 使用正则表达式去除非文本符号（保留字母、数字、空格和标点符号）
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip() # 可选：移除多余的空格
        self.text = cleaned_text
        self.words =word_tokenize(self.text)
        # self.sentences = sent_tokenize(self.text)
        
        self.stop_words = set(stopwords.words('english'))
        self.nlp = nlp
        self.doc = self.nlp(self.text)
        self.sentences = [sent.text for sent in self.doc.sents]
        self.word_count = len(self.words)
        self.sent_count = len(self.sentences)


class ReviewSim:

    def __init__(self, Reviews):
        self.Reviews = Reviews
        self.ReviewSim = self.get_sim()

    
    def get_sim(self):

        paper_review_sim = []

        for i, current_review in enumerate(self.Reviews):
            for j, compare_review in enumerate(self.Reviews):
                if i!=j:
                    review_sim = cosine_similarity(current_review.reshape(1, -1), compare_review.reshape(1, -1))
                    paper_review_sim.append(review_sim)
        review_score = np.max(paper_review_sim)
        return review_score
    


class MetaReviewSim:
    def __init__(self, Meta, Reviews):
        self.Meta = Meta
        self.Reviews = Reviews
        self.meta_review_sim = self.get_sim()

    
    def get_sim(self):
        paper_review_sim = []

        for i, current_review in enumerate(self.Reviews):
            review_sim = cosine_similarity(self.Meta.reshape(1, -1), current_review.reshape(1, -1))
            paper_review_sim.append(review_sim)
        review_score = np.mean(paper_review_sim)
        return review_score
    

class ReviewSFIRF:
    def __init__(self, Reviews, threshold=0.5):
        self.Reviews = Reviews
        self.threshold = threshold
        self.SFIRF = self.get_SFIRF()

    
    def get_SFIRF(self):
        paper_sent_SF = []
        paper_sent_IRF = []
        paper_sent_SF_IRF = []


        for i, current_review in enumerate(self.Reviews):
            current_review_sent_sim = np.vstack(cosine_similarity(current_review)) # 同一条review中，不同sent与其他sent的相似度
            current_review_sent_sim[current_review_sent_sim < self.threshold] = 0
            SF_value = np.mean(current_review_sent_sim, axis=1)  # 当前review的每个句子的SF
            IRF_max_sim = []
            for j, compare_review in enumerate(self.Reviews):
                if i!=j:
                    compare_review_sent_sim = np.vstack(cosine_similarity(current_review, compare_review))
                    compare_review_sent_sim[compare_review_sent_sim < self.threshold] = 0
                    max_sim = np.max(compare_review_sent_sim, axis=1)
                    IRF_max_sim.append(max_sim)
            IRF_value = np.log(len(self.Reviews)/(np.sum(np.vstack(IRF_max_sim), axis=0)+0.1)) # [sen_len, ]
            SF_IRF_value = SF_value*IRF_value # SF-IRF(s, r, R) # [sen_len, ]

            paper_sent_SF.append(np.mean(SF_value))
            paper_sent_IRF.append(np.mean(IRF_value))
            paper_sent_SF_IRF.append(np.mean(SF_IRF_value))

        SFIRF = {
            'SF': np.mean(paper_sent_SF),
            'IRF': np.mean(paper_sent_IRF),
            'SF_IRF': np.mean(paper_sent_SF_IRF)
        }
        return SFIRF
    
class MetaReviewSFIRF:
    def __init__(self, Meta, Reviews, threshold=0.5):
        self.Meta = Meta
        self.Reviews = Reviews
        self.threshold = threshold
        self.SFIRF = self.get_SFIRF()

    
    def get_SFIRF(self):
        current_review_sent_sim = np.vstack(cosine_similarity(self.Meta)) 
        current_review_sent_sim[current_review_sent_sim < self.threshold] = 0
        SF_value = np.mean(current_review_sent_sim, axis=1)  
        IRF_max_sim = []
        for compare_review in self.Reviews:
            compare_review_sent_sim = np.vstack(cosine_similarity(self.Meta, compare_review))
            compare_review_sent_sim[compare_review_sent_sim < self.threshold] = 0
            max_sim = np.max(compare_review_sent_sim, axis=1)
            IRF_max_sim.append(max_sim)
        IRF_value = np.log(len(self.Reviews)/(np.sum(np.vstack(IRF_max_sim), axis=0)+0.1)) # [sen_len, ]
        SF_IRF_value = SF_value*IRF_value 


        SFIRF = {
            'SF': np.mean(SF_value),
            'IRF': np.mean(IRF_value),
            'SF_IRF': np.mean(SF_IRF_value)
        }
        return SFIRF
    
