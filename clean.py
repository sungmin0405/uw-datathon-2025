import pandas as pd
import os # Import the os module for path joining
import sys # Import sys to handle potential script execution contexts

def clean_csv_data(input_path, output_path, expected_cols):
    """
    Reads a CSV file, removes cancelled orders (InvoiceNo starts with 'C'),
    removes rows with missing values in any column, and saves the cleaned
    data to a new CSV file.

    Args:
        input_path (str): The full path to the input CSV file.
        output_path (str): The full path where the cleaned CSV file will be saved.
        expected_cols (list): A list of expected column names for validation.
    """
    print(f"Attempting to read input file: {input_path}")

    try:
        # Read the CSV file into a pandas DataFrame
        # Specify the encoding, often 'ISO-8859-1' or 'latin1' for retail datasets
        # Use `na_values=['']` to treat empty strings as missing values (NaN)
        # Use `low_memory=False` to prevent potential dtype guessing issues with large files
        df = pd.read_csv(input_path, encoding='ISO-8859-1', na_values=[''], low_memory=False)

        print(f"Successfully read {input_path}.")
        print(f"Original data shape (rows, columns): {df.shape}")
        print(f"Original columns: {df.columns.tolist()}")

        # Optional: Check if all expected columns are present
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: The following expected columns are missing from the CSV: {missing_cols}")
            # For now, we'll continue with the columns that are present.

        # --- Data Cleaning Step 1: Remove Cancelled Orders ---
        # Ensure InvoiceNo is treated as string before checking startswith
        # Handle potential NaN values in InvoiceNo gracefully by setting na=False
        # The ~ negates the condition, keeping rows that DO NOT start with 'C'
        original_rows = df.shape[0]
        if 'InvoiceNo' in df.columns:
            # Convert InvoiceNo to string type first to avoid errors with non-string data
            df['InvoiceNo'] = df['InvoiceNo'].astype(str)
            df = df[~df['InvoiceNo'].str.startswith('C', na=False)]
            rows_after_cancellation_removal = df.shape[0]
            cancelled_removed = original_rows - rows_after_cancellation_removal
            print(f"Removed {cancelled_removed} rows for cancelled orders (InvoiceNo starting with 'C').")
            print(f"Shape after removing cancellations: {df.shape}")
        else:
            print("Warning: 'InvoiceNo' column not found. Skipping cancellation removal step.")


        # --- Data Cleaning Step 2: Remove Rows with Missing Values ---
        rows_before_na_drop = df.shape[0]
        # Drop rows where ANY column has a missing value (NaN)
        # `dropna()` by default drops rows if any value in the row is NaN.
        df_cleaned = df.dropna(how='any')
        rows_after_na_drop = df_cleaned.shape[0]
        na_removed = rows_before_na_drop - rows_after_na_drop

        print(f"Removed {na_removed} rows due to missing values in one or more columns.")
        print(f"Final cleaned data shape (rows, columns): {df_cleaned.shape}")

        total_rows_removed = original_rows - df_cleaned.shape[0]
        print(f"Total number of rows removed: {total_rows_removed}")


        # --- Save Cleaned Data ---
        # Save the cleaned DataFrame to a new CSV file
        # `index=False` prevents pandas from writing the DataFrame index as a column
        # Use 'utf-8' encoding for the output file for better compatibility
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8')

        print(f"Cleaned data successfully saved to '{output_path}'")
        return True # Indicate success

    except FileNotFoundError:
        print(f"Error: The input file was not found at '{input_path}'.")
        print("Please ensure the file exists and the path is correct.")
        return False # Indicate failure
    except pd.errors.EmptyDataError:
        print(f"Error: The input file '{input_path}' is empty.")
        return False # Indicate failure
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        # Consider more specific error handling if needed
        return False # Indicate failure

def main():
    """
    Main function to configure paths and call the cleaning function.
    """
    # --- Configuration ---
    # Determine the script's directory safely
    if getattr(sys, 'frozen', False):
        script_dir = os.path.dirname(sys.executable)
    else:
        try:
           script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
           script_dir = '.' # Use current working directory

    input_csv_filename = os.path.join(script_dir, '/Users/eubenein/Desktop/projs/Datathon/Mode_Craft_Ecommerce_Data - Online_Retail.csv')  # <- IMPORTANT: Update 'your_data.csv'
    output_csv_filename = os.path.join(script_dir, 'cleaned_data.csv')

    expected_columns = [
        'InvoiceNo', 'StockCode', 'Description', 'Quantity',
        'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'
    ]

    # --- Execute Cleaning ---
    print("--- Starting Data Cleaning Process ---")
    success = clean_csv_data(input_csv_filename, output_csv_filename, expected_columns)
    print("--- Data Cleaning Process Finished ---")

    if success:
        print("\nScript finished successfully.")
    else:
        print("\nScript finished with errors.")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()
