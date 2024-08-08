import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

def load_data():
    # Specify the path to the dataset
    dataset_path = 'phishing_site_urls.csv'
    data = pd.read_csv(dataset_path)
    return data

def preprocess_data(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['URL'])
    y = data['Label'].apply(lambda x: 1 if x == 'bad' else 0)  # Convert labels to binary
    return X, y, vectorizer

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, predictions)}')
    return model

def save_model(model, vectorizer):
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

if __name__ == "__main__":
    data = load_data()
    X, y, vectorizer = preprocess_data(data)
    model = train_model(X, y)
    save_model(model, vectorizer)

