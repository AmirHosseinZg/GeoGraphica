import pandas as pd
from decimal import Decimal, getcontext


def data_inserter(list_data):
    # Sample dictionary
    data = list_data

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    df.to_excel("Txx_calculated_data_example.xlsx", index=False)
    print("the excel file created :)")


def data_retriever(file_path):
    # Set precision for Decimal
    getcontext().prec = 50

    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path, header=None)

    # Convert DataFrame to 2D list with Decimal type
    data = df.applymap(lambda x: Decimal(x)).values.tolist()

    return data
