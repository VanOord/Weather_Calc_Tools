# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:31:26 2023

@author: q4r
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar
import PySimpleGUI as sg

def infer_date_format(date_string):
    formats = [
        '%Y-%m-%d %H:%M',
        '%d-%m-%Y %H:%M',
        '%m-%d-%Y %H:%M',
        '%d/%m/%Y %H:%M',
        '%m/%d/%Y %H:%M',
        '%Y/%m/%d %H:%M',
    ]
    for fmt in formats:
        try:
            if fmt in ['%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M']:
                pd.to_datetime(date_string, format=fmt, dayfirst=False)
            else:
                pd.to_datetime(date_string, format=fmt, dayfirst=True)
            return fmt
        except ValueError:
            continue
    return None

def find_data_start(file_path):
    with open(file_path) as file:
        for line_num, line in enumerate(file):
            row_data = line.strip().split(",")
            if len(row_data) >= 2:
                return line_num
    return None

def plot_csv_data(file_path, time, investigation):
    # Determine the start of the data dynamically
    data_start = find_data_start(file_path)
    if data_start is None:
        print("Data start not found.")
        return

    # Load the data from the CSV file, skipping the header lines
    df = pd.read_csv(file_path, skiprows=data_start)
    # Create a list of header names from the CSV file
    header_names = list(df.columns)
    
    # Define the layout for the GUI window
    layout = [
        [sg.Text("Choose date/time column: "), sg.Combo(header_names, default_value=time, key='time')],
        [sg.Text('')],
        [sg.Text("Choose column to investigate: "), sg.Combo(header_names, default_value=investigation, key='investigation')],
        [sg.Text('')],
        [sg.Text('Enter the desired limit:')],
        [sg.Input(key='-LIMIT-')],
        [sg.Text('')],
        [sg.Text('Enter the weather window duration in hours (enter "1" for default timestep of 1 hour):')],
        [sg.Input(key='-WINDOW SIZE-')],
        [sg.Text('')],
        [sg.Text('Do you want to investigate persistency based on certain months?')],
        [sg.Radio('Yes', 'RADIO1', key='-RADIO YES-'), sg.Radio('No', 'RADIO1', key='-RADIO NO-', default=True)],
        [sg.Text('')],
        [sg.Text('Start month:'), sg.Combo(list(calendar.month_abbr[1:]), key='-START MONTH-', enable_events=True, disabled=False)],
        [sg.Text('End month:'), sg.Combo(list(calendar.month_abbr[1:]), key='-END MONTH-', enable_events=True, disabled=False)],
        [sg.Text('')],
        [sg.Button("Plot"), sg.Button("Exit")]
    ]
    
    # Create the GUI window
    window = sg.Window("CSV Plotter", layout)
    
    # Event loop to process GUI events
    while True:
        event, values = window.read()
        
        if event == "Plot":
            time = values['time']
            investigation = values['investigation']
            inputted_limit = float(values['-LIMIT-'])
            window_size = int(values['-WINDOW SIZE-'])
            investigate_range = values['-RADIO YES-']
            start_month = values['-START MONTH-']
            end_month = values['-END MONTH-']
            window.close()
            break
        
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
    
    # Close the GUI window
    window.close()
    
    # Perform the plotting
    first_value = str(df[time].iloc[0])
    #print(f"First value: {first_value}") # Can print in console for debugging purposes
    # Infer the date format based on the first value
    date_format = infer_date_format(first_value)
    #print(f"Inferred date format: {date_format}") # Can print in console for debugging purposes
    
    if date_format is None:
        print("Unable to infer date format.")
        return

    # Convert the time column to a datetime format with the inferred format
    df[time] = pd.to_datetime(df[time], format=date_format)
    # Set the datetime column as the index
    df.set_index(time, inplace=True)
    
    # Resample the data to hourly frequency and select the mean value for each hour
    df = df.resample('H').mean()
    #print(df[264:290])
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Filter the data based on the selected months
    if investigate_range:
        # Convert month names to month numbers
        month_dict = {v.lower(): k for k, v in enumerate(calendar.month_abbr) if k != 0}
        start_month_num = month_dict[start_month.lower()]
        end_month_num = month_dict[end_month.lower()]
        selected_months = range(start_month_num, end_month_num + 1)

        # Filter the data based on the selected months
        df_filtered = df[df.index.month.isin(selected_months)]

        # Calculate persistency for the selected month range
        values = df_filtered[investigation].values
        num_windows = len(values) - window_size + 1
        count = np.sum(np.all(values[np.arange(window_size)[:, None] + np.arange(num_windows)] < inputted_limit, axis=0))
        avg_workability = (count / len(values)) * 100
        additional_info = f"Average Persistency: {avg_workability:.2f}% for months {start_month} to {end_month} with weather window of {window_size} hour(s) and limit of {inputted_limit}"
        print(additional_info)
        print()
        # Plot the data for the selected month range
        ax.plot(df_filtered.index, df_filtered[investigation], color='tab:blue', linewidth=0.15)

        # Add text box with persistency value
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        textstr = f"Average Persistency: {avg_workability:.2f}% for months {start_month} to {end_month} with weather window of {window_size} hour(s) and limit of {inputted_limit}"
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, color='black')
    
    else:
        # Calculate persistency for the entire data range
        values = df[investigation].values
        num_windows = len(values) - window_size + 1
        count = np.sum(np.all(values[np.arange(window_size)[:, None] + np.arange(num_windows)] < inputted_limit, axis=0))
        workability = (count / len(values)) * 100
        additional_info = f"Persistency: {workability:.2f}% with a weather window of {window_size} hour(s) and a limit of {investigation} = {inputted_limit}"
        print(additional_info)
        print()
    
        # Plot the data for the entire range
        ax.plot(df.index, df[investigation], color='tab:blue', linewidth=0.15)
    
        # Add text box with persistency value
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        textstr = f"Persistency: {workability:.2f}% with a weather window of {window_size} hour(s) and a limit of {inputted_limit}"
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, color='black')
        
    # PRINT MONTHLY VALUES
    start_month_num = 1
    end_month_num = 12
    selected_months_W = range(start_month_num, end_month_num + 1)
    selected_month_names_W = [calendar.month_name[i] for i in selected_months_W]
    # Initialize a list to store persistency values for the selected months
    month_workability_all = []

    # Iterate over the selected months
    for month in selected_months_W:
        # Filter the data for the current month
        month_data = df[df.index.month == month]

        # Calculate persistency for the current month
        values = month_data[investigation].values
        num_windows = len(values) - window_size + 1
        count = np.sum(np.all(values[np.arange(window_size)[:, None] + np.arange(num_windows)] < inputted_limit, axis=0))
        workability_month = (count / len(values)) * 100
        month_workability_all.append(workability_month)

    # Create a DataFrame containing the month and mean persistency
    table_data = pd.DataFrame({"Month": selected_month_names_W, "Mean Persistency": month_workability_all})
    # Format the values in the "Mean persistency" column to two decimal places
    table_data["Mean Persistency"] = table_data["Mean Persistency"].map("{:.2f}%".format)
    # Print the table_data DataFrame without the index column
    print(table_data.to_string(index=False))

    # Export the DataFrame to a CSV file without the index column
    output_file_path = "monthly_persistency.csv"
    table_data.to_csv(output_file_path, index=False)

    # Read the contents of the CSV file
    with open(output_file_path, 'r') as f:
        csv_contents = f.read()

    # Concatenate the console_output, a newline character, a space, and the table data
    output_contents = f"{additional_info}\n\n{csv_contents}"

    # Write the modified contents back to the file
    with open(output_file_path, 'w') as f:
        f.write(output_contents)

    print()
    print(f"Monthly mean persistency table has been exported to '{output_file_path}' CSV file.")
    print()
    
    # Plot line
    ax.axhline(y=inputted_limit, color='red', linestyle='--', linewidth=1)

    # Set the x-axis label
    ax.set_xlabel('Date')

    # Set the y-axis label
    ax.set_ylabel(investigation)

    # Set the title
    ax.set_title(f'{investigation} over Time')

    # Show grid lines
    ax.grid(True)

    # Define dictionary of frequency to format string mappings
    freq_fmt_dict = {
        'AS': '%Y',
        '2AS': '%Y'
    }
    # Determine default frequency based on number of data points
    num_points = len(df) / 24

    if num_points <= 3653:
        freq = 'AS'
        freq_fmt = freq_fmt_dict[freq]
    else:
        freq = '2AS'
        freq_fmt = freq_fmt_dict[freq]

    xticks = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    xticks = xticks[xticks >= pd.to_datetime(df.index.min())]
    xticks_shifted = xticks.shift(1, freq=freq)
    xticks = xticks.union(xticks_shifted)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks.strftime(freq_fmt))

    # Rotate x-axis labels to avoid overlapping
    plt.xticks(rotation=45)
    #save the plot
    plt.savefig('figure_persistency.png', dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()

# Define the layout for the initial GUI window
layout = [
    [sg.Text("Enter the file path for the CSV file: "), sg.InputText(), sg.FileBrowse()],
    [sg.Button("Next"), sg.Button("Exit")]
]

# Create the initial GUI window
window = sg.Window("CSV Plotter", layout)

# Event loop to process GUI events
while True:
    event, values = window.read()

    if event == "Next":
        file_path = values[0]
        window.close()
        break

    if event in (sg.WINDOW_CLOSED, "Exit"):
        break

# Close the initial GUI window
window.close()

# Open the next GUI window to choose columns and plot the data
plot_csv_data(file_path, "", "")
