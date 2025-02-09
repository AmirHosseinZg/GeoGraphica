import csv

# Initialize the data dictionary
data = {}

# Specify the path to your CSV file
csv_file_path = "../../database/EGM96.csv"

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
            C_nm = float(row[3])  # Third column: C_nm (should be float)
            S_nm = float(row[4])  # Fourth column: S_nm (should be float)

            # Initialize nested dictionaries if needed
            if n not in data:
                data[n] = {}
            data[n][m] = [C_nm, S_nm]

        except ValueError as e:
            print(f"Error processing row {row}: {e}")

C20 = -0.484169650276 * 10**-3
C40 = 0.790314704521 * 10**-6
C60 = -0.168729437964 * 10**-8
C80 = 0.346071647263 * 10**-11
C100 = -0.265086254269 * 10**-14

data[2][0][0] = data[2][0][0] - C20
data[4][0][0] = data[4][0][0] - C40
data[6][0][0] = data[6][0][0] - C60
data[8][0][0] = data[8][0][0] - C80
data[10][0][0] = data[10][0][0] - C100
# Now `data` dictionary contains your EGM96 coefficients with (n, m) as keys
