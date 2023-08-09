#Finished on Aug. 9th, 2023
#Author q4r, propery Van Oord Offshore Wind

import pandas as pd
import numpy as np
import calendar
import PySimpleGUI as sg
from dateutil.parser import parse

# Defining functions

def find_data_start(file_path):
    with open(file_path) as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        row_data = line.strip().split(",")
        if len(row_data) >= 2:
            try:
                parse(row_data[0])
                if all(len(line.strip().split(",")) >= 2 for line in lines[i+1:i+21]):
                    return i - 1  # Adjusted to return the index of the header line
            except ValueError:
                continue

    return None

def calculate_persistency_multi(values, window_size, inputted_limits, values_op):
    
    assert len(values) == len(inputted_limits), "Mismatch in the number of conditions"

    num_windows = len(values[0]) - window_size + 1
    threshold_met_windows = np.ones((num_windows, ), dtype=bool)
    
    for val, limit in zip(values, inputted_limits):
        if val is None: # Will skip investigation2 and 3 if they are left blank
            continue
        window_values = val[np.arange(window_size)[:, None] + np.arange(num_windows)]
        if values_op == '-RADIO BELOW-':
            condition_windows = window_values < limit
        else:
            condition_windows = window_values > limit
        
        threshold_met_windows &= np.all(condition_windows, axis=0)
    
    count = np.sum(threshold_met_windows) # This value gives the actual number of overlapping weather windows 
    persistency = (count / len(values[0])) * 100 # Gives persistency percentage
    
    return persistency, count

def calculate_persistency_non_overlapping_multi(values, window_size, inputted_limits, values_op):
    
    assert len(values) == len(inputted_limits), "Mismatch in the number of conditions"
    
    if len(values[0]) == 0:
        return 0, 0  # Avoid division by zero and return zeros if values list is empty

    count = 0
    i = 0
    
    while i <= len(values[0]) - window_size:
        all_conditions_met = True  # Initial assumption
        
        for val, limit in zip(values, inputted_limits):
            if val is None: # Will skip investigation2 and 3 if they are left blank
                continue
            window = val[i:i+window_size]
            
            if values_op == '-RADIO BELOW-':
                condition_met = np.all(window < limit)
            else:
                condition_met = np.all(window > limit)
            
            all_conditions_met &= condition_met  # If any condition is not met, this becomes False
        
        if all_conditions_met:
            count += 1 #This value, when the loop is completed, is all the non over-lapping weather windows
            i += window_size  # Move to the end of this window
        else:
            i += 1  # Move one step forward

    persistency = count / len(values[0]) * 100 #Persistency percentage
    return persistency, count

def main_calculations(file_path, time, investigation1, investigation2, investigation3):
    
    csv_messages = [] #create csv file message dictionary
    output_messages = [] #create GUI output message dictionary

    # Determine the start of the data based on if there is a datetime format atleast 20 rows long in the first column
    data_start = find_data_start(file_path)
    if data_start is None:
        output_messages.append("Data start not found.")
        return

    # Load chunk of data from the CSV file to get the header names of the columns
    chunk_df = pd.read_csv(file_path, nrows=100, skiprows=data_start)
    time = chunk_df.columns[0] #will be the index column, CODE ONLY WORKS IF DATETIME COLUMN IS FIRST
    header_names1 = list(chunk_df.columns[1:]) #Skips first column header (time)
    header_names2 = list(chunk_df.columns[2:]) #Skips first two column headers
    header_names3 = list(chunk_df.columns[3:]) #Skips first three column headers
    # Define the layout for the GUI window
    layout = [
        [sg.Text("Choose column(s) to investigate: ")],
        [sg.Text("Column 1 (required): "), sg.Combo(header_names1, default_value=investigation1, key='investigation1'), sg.Text("Limit: "), sg.Input(key='-LIMIT1-')],
        [sg.Text("Column 2 (optional): "), sg.Combo(header_names2, key='investigation2'), sg.Text("Limit: "), sg.Input(key='-LIMIT2-')],
        [sg.Text("Column 3 (optional): "), sg.Combo(header_names3, key='investigation3'), sg.Text("Limit: "), sg.Input(key='-LIMIT3-')],
        #[sg.Text('')], #can add a spaced line here if wanted
        #Other definitions
        [sg.Radio('Calculate below limit', 'RADIO', key='-RADIO BELOW-', default=True),
         sg.Radio('Calculate above limit', 'RADIO', key='-RADIO ABOVE-')],
        [sg.Text('Enter the weather window duration in hours (enter "1" for default timestep of 1 hour:')],
        [sg.Input(key='-WINDOW SIZE-')],
        #[sg.Text('')],
        [sg.Text('Weather Window Type:')],
        [sg.Radio('Overlapping', 'RADIO3', key='-RADIO overlapping-', default=True),
         sg.Radio('Non-overlapping', 'RADIO3', key='-RADIO NON overlapping-')],
        #[sg.Text('')],
        [sg.Text('Do you want to investigate persistency based on certain months?')],
        [sg.Radio('No', 'RADIO1', key='-RADIO NO-', default=True), sg.Radio('Yes', 'RADIO1', key='-RADIO YES-')],
        #[sg.Text('')],
        [sg.Text('              Start month:'), sg.Combo(list(calendar.month_name[1:]), key='-START MONTH-', enable_events=True, disabled=False)],
        [sg.Text('              End month:'), sg.Combo(list(calendar.month_name[1:]), key='-END MONTH-', enable_events=True, disabled=False)],
        #[sg.Text('')],
        [sg.Button("Plot"), sg.Button("Exit")]
    ]

    # Create the GUI window
    window = sg.Window("Persistency Tool", layout)

    # Event loop to process GUI events
    while True:
        event, values = window.read()

        if event == "Plot":
            investigation1 = values['investigation1']
            limit1 = float(values['-LIMIT1-'])
            investigation2 = values['investigation2']
            limit2 = float(values['-LIMIT2-']) if values['-LIMIT2-'] else None
            investigation3 = values['investigation3'] 
            limit3 = float(values['-LIMIT3-']) if values['-LIMIT3-'] else None
            values_op = '-RADIO BELOW-' if values['-RADIO BELOW-'] else '-RADIO ABOVE-'
            window_size = int(values['-WINDOW SIZE-'])
            overlapping_WW = values['-RADIO overlapping-']
            investigate_range = values['-RADIO YES-']
            start_month = values['-START MONTH-']
            end_month = values['-END MONTH-']
            window.close()
            break

        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

    # Close the GUI window
    window.close()
    
    # Create a list of columns to load based on user's input
    columns_to_load = [time, investigation1]
    if investigation2:
        columns_to_load.append(investigation2)
    if investigation3:
        columns_to_load.append(investigation3)
    
    #Load all the chosen data
    df = pd.read_csv(file_path, skiprows=data_start, usecols=columns_to_load, parse_dates=[time], dayfirst=True)
    #Set Index columnn as the first column
    df.set_index(time, inplace=True)
    #Resample data so it is hourly (really only matters if you are using data which is sampled at a quicker rate)
    df = df.resample('H').mean()
    
    # Resampling the data by month for the mean count in the non-overlapping for all data case
    monthly_counts_nonfiltered = df.resample('M').size()

     # Wiggle room in hours (still will consider a month full even if it is missing upto 13 hours)
    wiggle_room = 13

    # Determine which months have a count close to the full number of hours for that month
    full_months_approx = monthly_counts_nonfiltered[monthly_counts_nonfiltered.index.map(lambda dt: calendar.monthrange(dt.year, dt.month)[1]*24 - wiggle_room) <= monthly_counts_nonfiltered]

    # Get the number of full months considering the wiggle room
    num_full_months_approx = len(full_months_approx)
    
    #calculation of persistency percentage and count
    if investigate_range: #this case is for if certain months want to be investigated
        month_dict = {v.lower(): k for k, v in enumerate(calendar.month_name) if k != 0}
        start_month_num = month_dict[start_month.lower()]
        end_month_num = month_dict[end_month.lower()]
        selected_months = range(start_month_num, end_month_num + 1)
        df_filtered = df[df.index.month.isin(selected_months)]
        values1 = df_filtered[investigation1].values
        values2 = df_filtered[investigation2].values if investigation2 else None
        values3 = df_filtered[investigation3].values if investigation3 else None
        values = [values1,values2,values3]
        limits = [limit1,limit2,limit3]
        years = df.index.year.unique()
        complete_ranges_count = 0

        for year in years:
            # Filter dataframe for the specific year
            df_year = df[df.index.year == year]
            # Check if all months in the selected range exist for that year
            if set(selected_months).issubset(set(df_year.index.month)):
                complete_ranges_count += 1

        if overlapping_WW:
            persistency, count = calculate_persistency_multi(values, window_size, limits, values_op)
            additional_info_parts = [f"Persistency: {persistency:.2f}% for months {start_month} through {end_month} with an overlapping weather window of {window_size} hour(s) and limit(s) of {investigation1} {('<' if values_op == '-RADIO BELOW-' else '>')} {limit1}"]
        else:
            persistency, count = calculate_persistency_non_overlapping_multi(values, window_size, limits, values_op)
            mean_count = count/complete_ranges_count #Mean number of WW occurrences per year
            additional_info_parts = [f"Persistency: {persistency:.2f}% for months {start_month} through {end_month} with a non-overlapping weather window of {window_size} hour(s) and limit(s) of {investigation1} {('<' if values_op == '-RADIO BELOW-' else '>')} {limit1}"]

        # Append information for investigation2 if it's not blank
        if investigation2:
            additional_info_parts.append(f"{investigation2} {('<' if values_op == '-RADIO BELOW-' else '>')} {limit2}")

        # Append information for investigation3 if it's not blank
        if investigation3:
            additional_info_parts.append(f"{investigation3} {('<' if values_op == '-RADIO BELOW-' else '>')} {limit3}")

        if overlapping_WW == False:
            # Append the additional provided content
            additional_info_parts.append("\n" + "\n" + f"Total number of weather window occurences in chosen range: {count:.2f}" + "\n"+ f"Mean number of weather window occurrences per chosen range: {mean_count:.2f} ")

        # Join the parts to create the full additional info string
        additional_info = "; ".join(additional_info_parts)

        csv_messages.append(additional_info)
        csv_messages.append("")
        output_messages.append(additional_info)
        output_messages.append("")


    else: #this case is for when all the data wants to be looked at
        values1 = df[investigation1].values
        values2 = df[investigation2].values if investigation2 else None
        values3 = df[investigation3].values if investigation3 else None
        values = [values1,values2,values3]
        limits = [limit1,limit2,limit3]

        if overlapping_WW:
            persistency, count = calculate_persistency_multi(values, window_size, limits, values_op)
            additional_info_parts = [f"Persistency: {persistency:.2f}% with an overlapping weather window of {window_size} hour(s) and limit(s) of {investigation1} {('<' if values_op == '-RADIO BELOW-' else '>')} {limit1}"]
        else:
            persistency, count = calculate_persistency_non_overlapping_multi(values, window_size, limits, values_op)
            #Mean number of occurences per year based on total number of counts by number of full months times 12
            mean_count = (count/num_full_months_approx)*12
            additional_info_parts = [f"Persistency: {persistency:.2f}% with a non-overlapping weather window of {window_size} hour(s) and limit(s) of {investigation1} {('<' if values_op == '-RADIO BELOW-' else '>')} {limit1}"]

        # Append information for investigation2 if it's not blank
        if investigation2:
            additional_info_parts.append(f"{investigation2} {('<' if values_op == '-RADIO BELOW-' else '>')} {limit2}")

        # Append information for investigation3 if it's not blank
        if investigation3:
            additional_info_parts.append(f"{investigation3} {('<' if values_op == '-RADIO BELOW-' else '>')} {limit3}")

        if overlapping_WW == False:
            # Append the additional provided content
            additional_info_parts.append("\n" + "\n" + f"Number of weather window occurences: {count}" + "\n"+ f"Mean number of weather window occurrences per full year: {mean_count:.2f}")

        # Join the parts to create the full additional info string
        additional_info = "; ".join(additional_info_parts)

        csv_messages.append(additional_info)
        csv_messages.append("")
        output_messages.append(additional_info)
        output_messages.append("")
    
    # Creation of tables and output file
    start_year = df.index.min().year
    end_year = df.index.max().year
    num_years = end_year - start_year + 1

    yearly_data = []

    # For storing the sum of monthly persistencies and counts across all years
    monthly_persistencies_sum = [0] * 12
    monthly_counts_sum = [0] * 12

    for year in range(start_year, end_year + 1):
        year_data = [year]
        monthly_persistencies = []
        monthly_counts = []
        for month in range(1, 13):  # Loop through all months
            month_data = df[(df.index.month == month) & (df.index.year == year)]
            values1 = month_data[investigation1].values
            values2 = month_data[investigation2].values if investigation2 else None
            values3 = month_data[investigation3].values if investigation3 else None
            values = [values1, values2, values3]
            limits = [limit1, limit2, limit3]

            if overlapping_WW:
                persistency, count = calculate_persistency_multi(values, window_size, limits, values_op)
            else:
                persistency, count = calculate_persistency_non_overlapping_multi(values, window_size, limits, values_op)

            monthly_persistencies.append(persistency)
            monthly_counts.append(count)
            
            # Add persistency and count to the sum for the respective month
            monthly_persistencies_sum[month - 1] += persistency
            monthly_counts_sum[month - 1] += count

        # Calculate the yearly average persistency
        yearly_average = sum(monthly_persistencies) / len(monthly_persistencies)

        # Append monthly data and yearly average persistency to the year's data
        year_data.extend(monthly_persistencies)
        year_data.append(yearly_average)
        yearly_data.append(year_data)

    # Compute the mean monthly persistencies and counts
    mean_monthly_persistencies = [persistency_sum / num_years for persistency_sum in monthly_persistencies_sum]
    mean_monthly_counts = [count_sum / num_years for count_sum in monthly_counts_sum]

    month_abbr = [calendar.month_abbr[i] for i in range(1, 13)]

    # Create the header and the rows for mean monthly persistencies and counts
    if overlapping_WW == False:
        header = ["Year"] + [calendar.month_abbr[i] for i in range(1, 13)] + ["Mean Yearly Persistency"]
        mean_persistencies_row = ["Mean Monthly Persistency"] + ["{:.2f}%".format(persistency) for persistency in mean_monthly_persistencies] + ["-"]
        mean_counts_row = ["Mean Monthly Count"] + ["{:.2f}".format(count) for count in mean_monthly_counts] + ["-"]
    else:
        header = ["Year"] + [calendar.month_abbr[i] for i in range(1, 13)] + ["Mean Yearly Persistency"]
        mean_persistencies_row = ["Mean Monthly Persistency"] + ["{:.2f}%".format(persistency) for persistency in mean_monthly_persistencies] + ["-"]

    # Add percentage sign to data and format
    formatted_data = []
    for row in yearly_data:
        formatted_row = [row[0]]  # Year
        formatted_row.extend(["{:.2f}%".format(val) for val in row[1:-1]])  # Monthly data
        formatted_row.append("{:.2f}%".format(row[-1]))  # Average
        formatted_data.append(formatted_row)

    # Add the rows for mean monthly persistencies and counts
    if overlapping_WW == False:
        formatted_data.append(mean_persistencies_row)
        formatted_data.append(mean_counts_row)
    else:
        formatted_data.append(mean_persistencies_row)

    # Convert the table to a DataFrame for saving
    table_df = pd.DataFrame([header] + formatted_data)

    # CSV data saving 
    output_file_path = "monthly_persistency.csv"
    table_df.to_csv(output_file_path, index=False, header=False)

    csv_messages.append(table_df)

     # Creation of table to print in GUI output window
    if overlapping_WW == False: #Only adding count values for the non-overlapping case
        table_data_console = pd.DataFrame({"Month": month_abbr, "Mean Persistency": mean_monthly_persistencies, "Mean WW Occurences": mean_monthly_counts})
        table_data_console["Mean Persistency"] = table_data_console["Mean Persistency"].map("{:.2f}%".format)
        table_data_console["Mean WW Occurences"] = table_data_console['Mean WW Occurences'].map("{:.2f}".format)
        # Left-align the "Month" column and right-align the "Mean Persistency" column
        table_data_console["Month"] = table_data_console["Month"].apply(lambda x: x.ljust(14))
        table_data_console["Mean Persistency"] = table_data_console["Mean Persistency"].apply(lambda x: x.ljust(30))
        # Construct the custom header
        header = "Month".ljust(13) + "Mean Persistency".ljust(24) + "Mean WW Occurences"
    else:
        # Creation of table to print in GUI output window
        table_data_console = pd.DataFrame({"Month": month_abbr, "Mean Persistency": mean_monthly_persistencies})
        table_data_console["Mean Persistency"] = table_data_console["Mean Persistency"].map("{:.2f}%".format)
        # Left-align the "Month" column and right-align the "Mean Persistency" column
        table_data_console["Month"] = table_data_console["Month"].apply(lambda x: x.ljust(14))
        # Construct the custom header
        header = "Month".ljust(13) + "Mean Persistency"#.ljust(20)

    # Construct the table body
    table_body = table_data_console.to_string(index=False, header=False)

    # Combine the header and table body
    table_string = header + "\n" + table_body
    
    output_messages.append(table_string)

    #Create and write csv_messages data to CSV file
    with open(output_file_path, 'r') as f:
        csv_contents = f.read()

    output_contents = f"{additional_info}\n\n{csv_contents}"

    with open(output_file_path, 'w') as f:
        f.write(output_contents)

    output_messages.append("")
    output_messages.append(f"Complete persistency table has been exported to '{output_file_path}' CSV file.")

    return output_messages
    #End of main_calculation function
   
## Running of code truly begins here

layout = [
    [sg.Text("Enter the file path for the CSV file (DHI format): "), sg.InputText(), sg.FileBrowse()],
    [sg.Button("Next"), sg.Button("Exit")]
]

window = sg.Window("Persistency Tool", layout)

while True:
    event, values = window.read()

    if event == "Next":
        file_path = values[0]
        window.close()
        break

    if event in (sg.WINDOW_CLOSED, "Exit"):
        break

window.close()

output_messages = main_calculations(file_path, "", "", "", "")

# Determine the number of lines and the maximum line length in the output
lines = "\n".join(output_messages).split("\n")
num_lines = len(lines)
max_line_length = max(len(line) for line in lines)

# Create a multiline widget with a dynamic size
output_window = sg.Window('Output', [[sg.Multiline("\n".join(output_messages), size=(max_line_length, num_lines))], [sg.Button('Exit Program')]])

#Create and open final GUI window
while True:
    event, values = output_window.read()
    if event == sg.WINDOW_CLOSED or event == 'Exit Program':
        break
output_window.close() #Program ends when final GUI window is closed