# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:33:29 2024

@author: sunny
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import TweetTokenizer
from scipy.sparse import hstack

# 假设数据集文件路径
DATASET_FP = "IMDB Dataset.csv"
IRONIC_DATASET_FP = "SemEval2018-T3-train-taskA.txt"

# 加载情感分类数据集
data = pd.read_csv(DATASET_FP)

# 预处理函数，简化版
def preprocess_text(text):
    return text.lower()  # 简化处理，实际应用更复杂的预处理

data['review'] = data['review'].apply(preprocess_text)

# 加载反讽检测数据集并预处理
def parse_dataset(fp):
    y = []
    corpus = []
    with open(fp, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"):  # 跳过首行
                line = line.rstrip()
                label, tweet = line.split("\t")[1], line.split("\t")[2]
                y.append(int(label))
                corpus.append(tweet)
    return corpus, y

irony_corpus, irony_y = parse_dataset(IRONIC_DATASET_FP)

# 特征提取
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words="english")
irony_features = vectorizer.fit_transform(irony_corpus)

# 训练反讽检测模型
irony_model = LogisticRegression()
irony_model.fit(irony_features, irony_y)

# 使用反讽模型对情感分类数据集进行预测
data['irony'] = irony_model.predict(vectorizer.transform(data['review']))

# 划分数据集
train_texts, test_texts, train_labels, test_labels, train_irony, test_irony = train_test_split(
    data['review'], data['sentiment'], data['irony'], test_size=0.2, random_state=42)

# 情感分类特征提取
sentiment_vectorizer = TfidfVectorizer()
train_features = sentiment_vectorizer.fit_transform(train_texts)
test_features = sentiment_vectorizer.transform(test_texts)

# 将反讽特征作为额外特征融合
train_features = hstack([train_features, np.array(train_irony)[:, None]])
test_features = hstack([test_features, np.array(test_irony)[:, None]])

# 训练情感分类模型
sentiment_model = LogisticRegression()
sentiment_model.fit(train_features, train_labels)

# 在测试集上进行预测
pred_labels = sentiment_model.predict(test_features)

# 计算准确率和其他评估指标
accuracy = accuracy_score(test_labels, pred_labels)
report = classification_report(test_labels, pred_labels)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
