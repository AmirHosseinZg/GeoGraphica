import csv

# Initialize the data dictionary
data = {}

# Specify the path to your CSV file
csv_file_path = "H:\\Zakeri\\Samadi_pr\\GeoGraphica\\Sources\\EGM96.csv"

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
            n = int(row[0])  # First column: n
            m = int(row[1])  # Second column: m
            C_nm = float(row[2])  # Third column: C_nm (should be float)
            S_nm = float(row[3])  # Fourth column: S_nm (should be float)

            # Initialize nested dictionaries if needed
            if n not in data:
                data[n] = {}
            data[n][m] = (C_nm, S_nm)

        except ValueError as e:
            print(f"Error processing row {row}: {e}")

# Now `data` dictionary contains your EGM96 coefficients with (n, m) as keys
