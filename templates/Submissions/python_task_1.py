import pandas as pd



#Question1 -Car MAtrix Generation
def generate_car_matrix(df):
    """TO generate the car matrix

    Args:
        df (Dataframe): Input dataframe

    Returns:
        Dataframe: Matrix Generation - Car column
    """
    pivot_df = df.pivot(index='id_1', columns='id_2', values='car')

    #fill Nan value as 0
    pivot_df = pivot_df.fillna(0)

    # Set diagonal values to 0
    for idx in pivot_df.index:
        pivot_df.loc[idx, idx] = 0

    # Reset the index if needed
    pivot_df.reset_index(inplace=True)

    return pivot_df

#Question2 -Car Type Count Calculation
def get_type_count(df):
    """Get the car type count

    Args:
        df (dataframe): Dataframe

    Returns:
        Dict:Count the car_type
    """
    conditions = [
    (df['car'] <= 15),
    (df['car'] > 15) & (df['car'] <= 25),
    (df['car'] > 25)
    ]

    labels = ['low', 'medium', 'high']

    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=labels, right=False)

    # Calculate the count of occurrences for each car_type category
    car_type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    car_type_counts = dict(sorted(car_type_counts.items()))
    return car_type_counts


#Question-3 - Bus Count Index Retrieval
def get_bus_indexes(df):
    """Get the bus column index

    Args:
        df (Dataframe): Bus column

    Returns:
        List:list of bus column index
    """
    mean_bus_value = df['bus'].mean()

    # Identify indices where the bus values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes


#Question 4 - Route Filtering
def filter_routes(df):
    """FIlter routes

    Args:
        df (Df): Truck column

    Returns:
        List: route of truck
    """
    # Calculate the mean value of the 'truck' column

    mean_truck_value = df['truck'].mean()

    # Identify values in the 'route' column where the average of 'truck' column is greater than 7
    selected_routes = df[df['truck'] > 7]['route'].tolist()

    # Remove duplicates from the list of selected routes
    selected_routes = list(set(selected_routes))

    # Sort the selected routes in ascending order
    selected_routes.sort()

    return selected_routes

#Question 5 Matrix Value Modification
def multiply_matrix(df):
    """Multiply matrix

    Args:
        df (Df): Modified matrix

    Returns:
        DF: Matrix
    """
    modified_df = df.copy()

    # Apply the specified logic to each value in the DataFrame
    modified_df = modified_df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round the values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df


#Question 6 - Time Check
def time_check(df):
    """Time check

    Args:
        df (DF): start and end time

    Returns:
        DF: Dataframe
    """
    # Combine startDay and startTime columns to create a new 'start_timestamp' column
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%A %H:%M:%S')

    # Combine endDay and endTime columns to create a new 'end_timestamp' column
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%A %H:%M:%S')

    # Create a new 'day_of_week' column to store the day of the week for each timestamp
    df['day_of_week'] = df['start_timestamp'].dt.day_name()

    # Set multi-index based on 'id' and 'id_2'
    df.set_index(['id', 'id_2'], inplace=True)

    # Create a mask to identify incorrect timestamps
    incorrect_timestamps_mask = (
        (df['start_timestamp'].dt.time != pd.Timestamp('00:00:00').time()) |
        (df['end_timestamp'].dt.time != pd.Timestamp('23:59:59').time())
    )

    # Create a mask to identify missing days of the week
    missing_days_mask = df.groupby(['id', 'id_2'])['day_of_week'].nunique() < 7

    # Combine the two masks to identify incorrect timestamps or missing days
    result_series = incorrect_timestamps_mask | missing_days_mask

    return result_series

#Input data from question1 to question5
data1= "D:\dataset-1.csv"
input_1_df =pd.read_csv(data1)

# Answer 1
modified_df = generate_car_matrix(input_1_df)

#Answer 2
car_count_dict = get_type_count(input_1_df)
print(car_count_dict)

#Answer 3
bus_idx_lst = get_bus_indexes(input_1_df)
print(bus_idx_lst)

#Answer 4
filter_rot_lst = filter_routes(input_1_df)
print(filter_rot_lst)

#Answer 5
modified_df_mul = multiply_matrix(modified_df)
print(modified_df)

#Answer 6
data2 = "D:\dataset-2.csv"
input_2_df = pd.read_csv(data2)

time_ck = time_check(input_2_df)
print(time_ck)



# mod = generate_car_matrix(input_df)
# print(multiply_matrix(mod))