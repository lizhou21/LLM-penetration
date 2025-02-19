import re
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from nltk.corpus import stopwords
import textstat
import spacy
from textblob import TextBlob
nltk.download('punkt')
nltk.download('stopwords')


class DetectMetrics:
    def __init__(self, text, nlp):
        # cleaned_text = re.sub(r'[^a-zA-Z0-9\s\.,!?;:()-]', '', text) # 使用正则表达式去除非文本符号（保留字母、数字、空格和标点符号）
        cleaned_text = text.replace('\n', ' ').replace('\r', '')
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
            calculated_words = [word for word in word_tokenize(sentence) if word.isalpha()]
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
        dependency_relations = set()

        # 对每个句子提取依存关系
        for sentence in self.doc.sents:
            for token in sentence:
                dependency_relations.add(token.dep_)

        # dependency_relations = [token.dep_ for token in self.doc if token.dep_ != "punct"]  # 排除标点符号
        # dep_types = set(dependency_relations)  
        return len(dependency_relations)
    
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
        total_clauses = 0

        # 对每个句子计算子句数目
        for sentence in self.doc.sents:
            clauses = [token for token in sentence if token.dep_ in ['ccomp', 'xcomp', 'advcl', 'relcl', 'acl']]  # 从句
            total_clauses += len(clauses)

        # 计算子句密度（平均每个句子中的子句数量）
        scd = total_clauses / len(self.sentences) if len(self.sentences) > 0 else 0
        return scd
        # if len(self.sentences) == 0:
        #     return 0
        # total_subordinate_clauses = 0

        # for sentence in self.doc.sents:
        #     for token in sentence:
        #         if token.dep_ in ["advcl", "ccomp", "xcomp", "relcl"]:  # 常见的从句依存关系  # 这个不太对
        #             total_subordinate_clauses += 1

        # return total_subordinate_clauses / len(self.sentences)


    
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
# 示例
# text = """
#     The quick brown fox jumps over the lazy dog. This is a simple sentence for testing purposes.
#     The purpose is to calculate the readability score of this text. 
#     Text readability helps in assessing the complexity of written content.
# """
# metrics = DetectMetrics(text)
# a=metrics.Average_Sentence_Length()
# print(metrics.get_metrics())
