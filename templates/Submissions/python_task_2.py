import pandas as pd
import numpy as np

def calculate_distance_matrix(df):
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    distances = {}

    for _, row in df.iterrows():
        start_location, end_location, distance = row[1], row[0], row[1]

        distances.setdefault((start_location, end_location), 0)
        distances[(start_location, end_location)] += distance

    locations = sorted(set(df[1]).union(df[1]))
    distance_matrix = pd.DataFrame(index=locations, columns=locations)

    for location1 in locations:
        for location2 in locations:
            cumulative_distance = distances.get((location1, location2), 0)
            cumulative_distance += distances.get((location2, location1), 0)

            if location1 == location2:
                cumulative_distance = 0

            distance_matrix.loc[location1, location2] = cumulative_distance

    distance_matrix = distance_matrix.add(distance_matrix.T, fill_value=0)

    return distance_matrix
data3 = "D:\MapUp-Data-Assessment-F\datasets\dataset-3.csv"
inputcsv = pd.read_csv(data3)
distance_matrix = calculate_distance_matrix(inputcsv)
print(distance_matrix)


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    df_reset = df.reset_index()
    unrolled_df = pd.melt(df_reset, id_vars='index', var_name='id_end', value_name='distance')
    unrolled_df.columns = ['id_start', 'id_end', 'distance']
    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]
    unrolled_df.reset_index(drop=True, inplace=True)

    return unrolled_df

# Example usage
# Assuming result_matrix is the DataFrame obtained from Question 1
unrolled_result = unroll_distance_matrix(result_matrix)
print(unrolled_result)



def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    reference_df = df[df['id_start'] == reference_id]
    average_distance = reference_df['distance'].mean()
    lower_threshold = average_distance - (0.1 * average_distance)
    upper_threshold = average_distance + (0.1 * average_distance)
    within_threshold_df = df[(df['id_start'] != reference_id) & (df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]
    sorted_values = sorted(within_threshold_df['id_start'].unique())
    result_df = pd.DataFrame({'id_start': sorted_values})

    return result_df
reference_id = 10
result_df = find_ids_within_ten_percentage_threshold(unrolled_result, reference_id)
print(result_df)

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    return df
