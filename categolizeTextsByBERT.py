#####################################
# TK210418　黒羽晟
#　BERTと決定技で文書分類するプログラム
# 関数がバラバラだからいつか整形する
#　WordやPDFをラベル名をつけたフォルダに入れるだけで学習データを作れる。
# できればGoogle DriveのAPIをつかってOCRをただ乗りするプログラムで、JPEGやスキャンpdfにも対応させたかった。
#####################################

from transformers import BertModel, BertJapaneseTokenizer
from docx import Document
from pdfminer.high_level import extract_text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump
import os
import glob
import torch
import numpy as np
import shutil
import fnmatch

def load_data(base_path):
    folders = [os.path.join(base_path, name) for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
    label_mapping = {folder: i for i, folder in enumerate(folders)}
    return folders, label_mapping


def extract_text_from_file(file_path):
    if file_path.endswith('.docx'):
        doc = Document(file_path)
        text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    elif file_path.endswith('.pdf'):
        text = extract_text(file_path)
    return text


def vectorize_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_output = outputs[0][0][0]
    return cls_output.numpy()


def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    return clf


def save_model(model, path='model'):
    dump(model, path)


def predict_label(doc_path, clf, tokenizer, model):
    text = extract_text_from_file(doc_path)
    cls_output = vectorize_text(text, tokenizer, model)
    predicted_label = clf.predict(cls_output.reshape(1, -1))
    return predicted_label


def copy_file_to_output(doc_path, output_folder):
    os.makedirs(os.path.join('output', output_folder), exist_ok=True)
    shutil.copy(doc_path, os.path.join('output', output_folder))


def get_all_files_in_folder(base_path):
    matches = []
    for root, dirnames, filenames in os.walk(base_path):
        for extensions in ['*.docx', '*.pdf']:
            for filename in fnmatch.filter(filenames, extensions):
                matches.append(os.path.join(root, filename))
    return matches

def classify_and_move_files(file_paths, clf, tokenizer, model, reverse_label_mapping):
    for file_path in file_paths:
        predicted_label = predict_label(file_path, clf, tokenizer, model)
        output_folder = reverse_label_mapping[predicted_label[0]]
        copy_file_to_output(file_path, output_folder)

def train_and_save_model(folders, label_mapping, tokenizer, model, model_path):
    data = []
    for folder in folders:
        for file_path in glob.glob(os.path.join(folder, '*')):
            text = extract_text_from_file(file_path)
            cls_output = vectorize_text(text, tokenizer, model)
            data.append((cls_output, label_mapping[folder]))

    features, labels = zip(*data)
    features = np.stack(features)
    labels = np.array(labels)

    clf = train_model(features, labels)
    dump(clf, model_path)

    return clf

def main():
    base_path = 'rowdata'
    target_path = 'target'
    model_path = './model/model'
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

    folders, label_mapping = load_data(base_path)
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    if os.path.exists(model_path):
        answer = input('既存のモデルが存在します。再学習しますか？ (y/n): ')
        if answer.lower() == 'n':
            clf = joblib.load(model_path)
        else:
            clf = train_and_save_model(folders, label_mapping, tokenizer, model, model_path)
    else:
        clf = train_and_save_model(folders, label_mapping, tokenizer, model, model_path)

    target_files = get_all_files_in_folder(target_path)
    classify_and_move_files(target_files, clf, tokenizer, model, reverse_label_mapping)

if __name__ == '__main__':
    main()

