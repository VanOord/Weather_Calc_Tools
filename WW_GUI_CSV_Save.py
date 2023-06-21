# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:31:26 2023

@author: q4r
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar
from tabulate import tabulate
import PySimpleGUI as sg

def infer_date_format(date_string):
    formats = ['%Y-%m-%d %H:%M', '%d-%m-%Y %H:%M']
    for fmt in formats:
        try:
            pd.to_datetime(date_string, format=fmt)
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
        [sg.Input(key='-WINDOW HOURS-')],
        [sg.Text('')],
        [sg.Text('Do you want to investigate workability based on certain months?')],
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
            window_hours = int(values['-WINDOW HOURS-'])
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
    print(f"Date format: {first_value}")
    # Infer the date format based on the first value
    date_format = infer_date_format(first_value)
    #print(date_format)
    #if date_format is None:
        #print("Unable to infer date format.")
        #return

     # Convert the time column to a datetime format with the inferred format
    df[time] = pd.to_datetime(df[time], format=date_format)
    
    # Set the datetime column as the index
    df.set_index(time, inplace=True)
    
    # Resample the data to hourly frequency and select the mean value for each hour
    df = df.resample('H').mean()
    #print(df[264:290])
    # Create the plot
    fig, ax = plt.subplots(figsize=(18, 6))
    
    # Filter the data based on the selected months
    if investigate_range:
        # Convert month names to month numbers
        month_dict = {v.lower(): k for k, v in enumerate(calendar.month_abbr) if k != 0}
    
        try:
            start_month_num = month_dict[start_month.lower()]
            end_month_num = month_dict[end_month.lower()]
            selected_months = range(start_month_num, end_month_num + 1)
    
            # Filter the data based on the selected months
            df_filtered = df[df.index.month.isin(selected_months)]
    
            # Calculate the window size based on the window hours
            window_size = window_hours
    
            # Calculate workability for the selected month range
            values = df_filtered[investigation].values
            avg_workability = np.mean(np.all(values[np.arange(window_size)[:, None] + np.arange(len(values) - window_size + 1)] < inputted_limit, axis=0)) * 100
    
            # Plot the data for the selected month range
            ax.plot(df_filtered.index, df_filtered[investigation], color='tab:blue', linewidth=0.15)
    
            # Add text box with workability value
            props = dict(boxstyle='square', facecolor='white', alpha=0.5)
            textstr = f"Average Workability: {avg_workability:.2f}% for months {start_month} to {end_month} with weather window of {window_hours} hour(s) and limit of {inputted_limit}"
            ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, color='black')
    
            console_output = []
            console_output.append('')
            console_output.append(f"Average Workability: {avg_workability:.2f}% for months {start_month} to {end_month} with weather window of {window_hours} hour(s) and limit of {inputted_limit}")
            # Print the average workability for the selected months
            print(f"Average Workability: {avg_workability:.2f}% for months {start_month} to {end_month} with weather window of {window_hours} hour(s) and limit of {inputted_limit}")
        except KeyError:
            print("Invalid month format. Please enter a valid capitalized month name or abbreviation.")
    else:
        # Calculate workability for the entire data range
        values = df[investigation].values
        window_size = window_hours
    
        workability = np.mean(np.all(values[np.arange(window_size)[:, None] + np.arange(len(values) - window_size + 1)] < inputted_limit, axis=0)) * 100
    
        # Plot the data for the entire range
        ax.plot(df.index, df[investigation], color='tab:blue', linewidth=0.15)
    
        # Add text box with workability value
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        textstr = f"Workability: {workability:.2f}% with a weather window of {window_hours} hour(s) and a limit of {inputted_limit}"
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, color='black')
    
        # Print the workability value
        print(f"Workability: {workability:.2f}% with a weather window of {window_hours} hour(s) and a limit of {investigation} = {inputted_limit}")
        # Print the workability value into the csv
        console_output = []
        console_output.append('')
        console_output.append(f"Workability: {workability:.2f}% with a weather window of {window_hours} hour(s) and a limit of {investigation} = {inputted_limit}")
        
    #Printing workability of each month
    start_month_num = 1
    end_month_num = 12
    selected_months_W = range(start_month_num, end_month_num + 1)
    selected_month_names_W = [calendar.month_name[i] for i in selected_months_W]
    # Initialize a list to store workability values for the selected months
    month_workability_all = []

    # Iterate over the selected months
    for month in selected_months_W:
        # Filter the data for the current month
        month_data = df[df.index.month == month]

        # Calculate workability for the current month
        values = month_data[investigation].values
        workability_month = np.mean(np.all(values[np.arange(window_size)[:, None] + np.arange(len(values) - window_size + 1)] < inputted_limit, axis=0)) * 100
        month_workability_all.append(workability_month)

    # Create a list of lists containing the month name and its corresponding workability value
    table_data = [[month_name, f"{workability_month:.2f}%"] for month_name, workability_month in zip(selected_month_names_W, month_workability_all)]
    console_output.append('')
    console_output.append(f'Table of monthly mean workability with a weather window of {window_hours} hour(s) and a limit of {investigation} = {inputted_limit}')
    # Print the table
    console_output.append('')
    console_output.append(tabulate(table_data, headers=["Month", "Mean Workability"]))
    print(' ')
    print(f'Table of monthly mean workability with a weather window of {window_hours} hour(s) and a limit of {investigation} = {inputted_limit}')
    # Print the table
    print(' ')
    print(tabulate(table_data, headers=["Month", "Mean Workability"]))  
    
    # Save the console output to a CSV file
    with open('Workability_output.csv', 'w') as file:
        for line in console_output:
            file.write(line + '\n')

    # Close the file
    file.close()
    print(' ')
    print('Data saved in Workability_output.csv file found in directory') 
    
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
    plt.savefig('figure_workability.png', dpi=300, bbox_inches='tight')
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

