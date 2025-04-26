import pandas as pd
import os
import sys

def enrich_data(input_path, output_path):
    """
    Reads a cleaned CSV file, adds derived features (Revenue, Date/Time components, etc.),
    and saves the enriched data to a new CSV file.

    Args:
        input_path (str): The full path to the cleaned input CSV file.
        output_path (str): The full path where the enriched CSV file will be saved.
    """
    print(f"Attempting to read cleaned input file: {input_path}")

    try:
        # Read the cleaned CSV file
        # Use 'utf-8' encoding as specified in the previous script's output
        df = pd.read_csv(input_path, encoding='utf-8')
        print(f"Successfully read {input_path}.")
        print(f"Cleaned data shape (rows, columns): {df.shape}")

        # --- Feature Engineering ---
        print("Starting feature engineering...")

        # 1. Convert InvoiceDate to datetime objects
        # Use errors='coerce' to turn unparseable dates into NaT (Not a Time)
        if 'InvoiceDate' in df.columns:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
            # Check if any dates failed to parse
            if df['InvoiceDate'].isnull().any():
                print("Warning: Some InvoiceDate values could not be parsed and were set to NaT.")
                # Optionally, handle NaT rows (e.g., drop them, fill them)
                # df = df.dropna(subset=['InvoiceDate']) # Example: drop rows with invalid dates
        else:
            print("Error: 'InvoiceDate' column not found. Cannot derive date/time features.")
            return False # Indicate failure as date features are crucial

        # 2. Ensure Quantity and UnitPrice are numeric
        # Use errors='coerce' to turn non-numeric values into NaN
        if 'Quantity' in df.columns:
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        else:
             print("Error: 'Quantity' column not found. Cannot calculate Revenue.")
             return False

        if 'UnitPrice' in df.columns:
             df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
        else:
             print("Error: 'UnitPrice' column not found. Cannot calculate Revenue.")
             return False

        # Handle potential NaNs introduced by coercion if necessary
        # Example: df = df.dropna(subset=['Quantity', 'UnitPrice'])

        # 3. Calculate Revenue
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        print("- Added 'Revenue' column.")

        # 4. Extract Date/Time Features (only if InvoiceDate is valid datetime)
        if pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
            df['Day of Week'] = df['InvoiceDate'].dt.day_name()
            print("- Added 'Day of Week' column.")
            df['Time of Day'] = df['InvoiceDate'].dt.strftime('%H:%M:%S') # Format as HH:MM:SS string
            print("- Added 'Time of Day' column.")
            # Monday=0, Sunday=6. Weekend is 5 (Saturday) or 6 (Sunday).
            df['Weekday or Weekend'] = df['InvoiceDate'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
            print("- Added 'Weekday or Weekend' column.")
            df['Quarter'] = df['InvoiceDate'].dt.quarter
            print("- Added 'Quarter' column.")
        else:
             print("Skipping date/time feature extraction due to invalid 'InvoiceDate' column type.")


        # 5. Add Product Name column (copying Description)
        if 'Description' in df.columns:
            df['Product Name'] = df['Description']
            print("- Added 'Product Name' column (copied from 'Description').")
        else:
            print("Warning: 'Description' column not found. Cannot create 'Product Name' column.")


        print("Feature engineering complete.")
        print(f"Enriched data shape (rows, columns): {df.shape}")
        print(f"Columns added: {df.columns.tolist()}") # Show final list of columns

        # --- Save Enriched Data ---
        # Save the enriched DataFrame to a new CSV file
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Enriched data successfully saved to '{output_path}'")
        return True # Indicate success

    except FileNotFoundError:
        print(f"Error: The input file was not found at '{input_path}'.")
        print("Please ensure the 'cleaned_data.csv' file exists in the correct directory.")
        return False # Indicate failure
    except pd.errors.EmptyDataError:
        print(f"Error: The input file '{input_path}' is empty.")
        return False # Indicate failure
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        return False # Indicate failure


def main():
    """
    Main function to configure paths and call the enriching function.
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

    # Input file is the output of the previous script
    input_csv_filename = os.path.join(script_dir, '/Users/eubenein/Desktop/projs/Datathon/cleaned_data.csv')
    # Define a new name for the output file with enriched data
    output_csv_filename = os.path.join(script_dir, 'enriched_data.csv')

    # --- Execute Enrichment ---
    print("--- Starting Data Enrichment Process ---")
    success = enrich_data(input_csv_filename, output_csv_filename)
    print("--- Data Enrichment Process Finished ---")

    if success:
        print("\nScript finished successfully.")
    else:
        print("\nScript finished with errors.")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()
