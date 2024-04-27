import pandas as pd


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