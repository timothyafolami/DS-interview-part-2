import pandas as pd
from utils import clean_data

# loading the dataset
data = pd.read_csv('../data/dataset.csv')

# dropping duplicates
data_1 = data.drop_duplicates().reset_index(drop=True)

# cleaning the data
data_2 = clean_data(data_1)

# dropping missing values based on the description of the data
data_3 = data_2.dropna(subset=['Description']).reset_index(drop=True)

# saving cleaned data
data_3.to_csv('../data/cleaned_data.csv', index=False)

# creating a dataframe with just the unique stock codes and it's descirptions
unique_stock_codes = data_3['StockCode'].unique()
unique_descriptions = data_3.groupby('StockCode').first()['Description'].values
unique_data = pd.DataFrame({'StockCode': unique_stock_codes, 'Description': unique_descriptions})


# Creating a text file with the product descriptions. 
with open('../data/unique_descriptions.txt', 'w') as file:
    for i, row in unique_data.iterrows():
        file.write(f"{row['Description']}\n")
