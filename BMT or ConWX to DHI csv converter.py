import csv
import PySimpleGUI as sg
from datetime import datetime

def process_row(row):
    # Extract the date and time components based on the user input
    datetime_components = list(map(int, row[:num_datetime_columns]))

    # Create a datetime object from the extracted components
    dt = datetime(*datetime_components)

    # Rearrange the row, placing the datetime column first and excluding the corresponding columns
    new_row = [dt.strftime("%d-%m-%Y %H:%M:%S")] + row[num_datetime_columns:]

    # Write the modified row to the output file
    writer.writerow(new_row)

# Create the PySimpleGUI window layout
layout = [
    [sg.Text("Select the input CSV file: "), sg.Input(key="-INFILE-"), sg.FileBrowse()],
    [sg.Text("Number of time component columns: (Default BMT is 6, default ConWX is 4)"), sg.Drop(values=[str(i) for i in range(1, 7)], key="-NUMCOLUMNS-")],
    [sg.Button("Process"), sg.Button("Cancel")]
]
print('This converter file will only work with BMT or ConWX files that are identical to their specified formats. Please remove any rows before the header line if you encounter an error using this tool.')
# Create the PySimpleGUI window
window = sg.Window("CSV Processing", layout)

# Event loop to process window events
while True:
    event, values = window.read()

    # Close the window if the user clicks the Cancel button or closes the window
    if event == sg.WINDOW_CLOSED or event == "Cancel":
        break

    # Process the input if the user clicks the Process button
    if event == "Process":
        # Get the input values
        input_file = values["-INFILE-"]
        num_datetime_columns = int(values["-NUMCOLUMNS-"])

        # Close the PySimpleGUI window
        window.close()

        # Open the input and output files
        with open(input_file, "r") as csv_file, open("Converted_BMT_or_ConWX_to_DHI.csv", "w", newline="") as output_file:
            reader = csv.reader(csv_file, delimiter=",")
            writer = csv.writer(output_file, delimiter=",")

            # Check if the first row contains "LON" or "LAT"
            first_row = next(reader)
            if "LON" not in first_row and "LAT" not in first_row:
                header = first_row
            else:
                # Skip the second line
                next(reader)
                # Use the third line as the headers
                header = next(reader)

            header = ["Datetime"] + header[num_datetime_columns:]  # Rearrange the header
            writer.writerow(header)

            # Check if the Row after the header needs to be skipped (fourth row in default BMT format)
            row_after_header = next(reader)
            if not row_after_header[0].isdigit():
                #print('Skipping row, extra header line detected')
                continue
            else:
                process_row(row_after_header)

            # Process each row in the input file
            for row in reader:
                process_row(row)

        break

# Close the PySimpleGUI window
window.close()

# Display the completion popup message
sg.Popup("Processing complete. Data saved to: Converted_BMT_or_ConWX_to_DHI.csv")