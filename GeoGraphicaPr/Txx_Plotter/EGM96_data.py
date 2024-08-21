import csv

# Initialize the data dictionary
data = {}

# Specify the path to your CSV file
csv_file_path = "D:\programming\Projects\GeoGraphica\Sources\EGM96.csv"

# Open the CSV file
with open(csv_file_path, mode='r', newline='') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Skip empty rows
        if not row:
            continue

        try:
            # Extract and convert data
            n = int(row[0])
            m = int(row[1])
            C_nm = int(row[2])
            S_nm = int(row[3])

            # Initialize nested dictionaries if needed
            if n not in data:
                data[n] = {}
            data[n][m] = (C_nm, S_nm)

        except ValueError as e:
            print(f"Error processing row {row}: {e}")
