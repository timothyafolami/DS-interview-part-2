import pandas as pd
import torch
import os


def clean_data(data):
    """
    This function cleans a pandas DataFrame by:
        - Removing special characters from InvoiceNo, StockCode, Description and Country.
        - Converting the Quantity and UnitPrice columns to numeric data types (assuming they represent numbers).
        - Standardizing the format of the InvoiceDate column (assuming YYYY-MM-DD format is desired).

    Args:
        data: A pandas DataFrame containing the dataset.

    Returns:
        A pandas DataFrame containing the cleaned data.
    """

    # cleanng the customer column
    data['Country'] = data['Country'].str.replace("XxY", "", regex=True).str.rstrip("☺️")

    # Define columns to clean special characters from
    cols_clean_special_chars = ['InvoiceNo', 'StockCode', 'Description']

    # Clean special characters using regular expressions
    for col in cols_clean_special_chars:
        data[col] = data[col].str.replace(r'[^\w\s]', '', regex=True)

    extra_clean = ['InvoiceNo', 'StockCode', 'CustomerID']
    # making sure there's only numbers in the columns
    for col in extra_clean:
        data[col] = data[col].str.replace(r'\D', '', regex=True)

    # Try converting Quantity and UnitPrice to numeric (handle errors)
    for col in ['Quantity', 'UnitPrice']:
        try:
            # Assuming data contains numbers with potential separators (".", "w")
            data[col] = pd.to_numeric(data[col].str.replace(r'[^\d\-+\.]', '', regex=True))
        except:
            print(f"Error converting column {col} to numeric, data may contain invalid formats.")

    # Standardize InvoiceDate format (assuming desired format is YYYY-MM-DD)
    try:
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%Y-%m-%d %H:%M:%S')
    except:
        print(f"Error parsing InvoiceDate, data may contain invalid formats.")
        

    return data


# Saving model
def save_model(model, path):
  """Saves the model's state dictionary to a file."""
  torch.save(model.state_dict(), path)



def get_most_recent_file(folder_path='./static'):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out directories from the list of files
    files = [file for file in files if os.path.isfile(os.path.join(folder_path, file))]
    
    # Sort files based on modification time
    files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
    
    # Return the most recent file
    if files:
        return os.path.join(folder_path, files[0])
    else:
        return None

# creating a get price function.
# This function takes in a list of items and then return the price of the items as a list
def get_price(items, data='./data/cleaned_data.csv'):
    """
    This function takes in a list of items and then return the price of the items as a list

    Args:
        data: A pandas DataFrame containing the dataset.
        items: A list of items to get the price for.

    Returns:
        A list of prices for the items.
    """
    data = pd.read_csv(data)
    prices = []
    for item in items:
        item_data = data[data['Description'] == item]
        price = item_data['UnitPrice'].values
        if len(price) > 0:
            prices.append(price[0])
        else:
            prices.append(None)
    return prices


# another one for stock code
def get_stock_code(items, data='./data/cleaned_data.csv'):
    """
    This function takes in a list of items and then return the stock code of the items as a list

    Args:
        data: A pandas DataFrame containing the dataset.
        items: A list of items to get the stock code for.

    Returns:
        A list of stock codes for the items.
    """
    data = pd.read_csv(data)
    stock_codes = []
    for item in items:
        item_data = data[data['Description'] == item]
        stock_code = item_data['StockCode'].values
        if len(stock_code) > 0:
            stock_codes.append(int(stock_code[0]))
        else:
            stock_codes.append(None)
    return stock_codes

# Then lastly for country
def get_country(items, data='./data/cleaned_data.csv'):
    """
    This function takes in a list of items and then return the country of the items as a list

    Args:
        data: A pandas DataFrame containing the dataset.
        items: A list of items to get the country for.

    Returns:
        A list of countries for the items.
    """
    data = pd.read_csv(data)
    countries = []
    for item in items:
        item_data = data[data['Description'] == item]
        country = item_data['Country'].values
        if len(country) > 0:
            countries.append(country[0])
        else:
            countries.append(None)
    return countries