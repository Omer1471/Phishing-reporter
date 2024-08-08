import pandas as pd

def load_data():
    # Specify the path to the dataset
    dataset_path = 'phishing_site_urls.csv'
    data = pd.read_csv(dataset_path)
    print(data.head())
    return data

if __name__ == "__main__":
    load_data()

