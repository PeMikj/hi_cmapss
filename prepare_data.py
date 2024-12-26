import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle  # For saving and loading data

# Constants
DATASET_NAMES = ['FD001', 'FD002', 'FD003', 'FD004']
TRAIN_DATA_FILES = [('./raw_data/train_{}.txt'.format(x), x) for x in DATASET_NAMES]
TEST_DATA_FILES = [('./raw_data/test_{}.txt'.format(x), x) for x in DATASET_NAMES]
TEST_DATA_RUL_FILES = [('./raw_data/RUL_{}.txt'.format(x), x) for x in DATASET_NAMES]

OP_SETTING_COLUMNS = ['op_setting_{}'.format(x) for x in range(1, 4)]
SENSOR_COLUMNS = ['sensor_{}'.format(x) for x in range(1, 22)]
SELECTED_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
                    'sensor_11', 'sensor_12', 'sensor_15']

def get_column_names():
    """Generate column names for the C-MAPSS dataset."""
    return ['unit', 'time_cycles'] + OP_SETTING_COLUMNS + SENSOR_COLUMNS

def read_data(filename):
    """
    Read C-MAPSS data from a file and return a pandas DataFrame.
    """
    col_names = get_column_names()
    return pd.read_csv(
        filename,
        sep='\s+',  # Handles space-separated values
        header=None,
        names=col_names
    )

def split_units_by_subset(data, subset_column, test_size=0.2):
    """
    Splits the data into two DataFrames by taking a percentage of units from each subset.
    """
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    subsets = data[subset_column].unique()

    for subset_name in subsets:
        subset_data = data[data[subset_column] == subset_name]
        unique_units = subset_data['unit'].unique()

        # Split units into train and test
        train_units, test_units = train_test_split(unique_units, test_size=test_size, random_state=42)

        # Append the corresponding data to train and test DataFrames
        train_data = pd.concat([train_data, subset_data[subset_data['unit'].isin(train_units)]])
        test_data = pd.concat([test_data, subset_data[subset_data['unit'].isin(test_units)]])

    # Sort the results
    train_data = train_data.sort_values(by=['unit', 'time_cycles']).reset_index(drop=True)
    test_data = test_data.sort_values(by=['unit', 'time_cycles']).reset_index(drop=True)
    
    return train_data, test_data

def scale_data(train_data, test_data, selected_columns):
    """
    Apply Min-Max scaling to the selected columns of the training and testing datasets.
    """
    scaler = MinMaxScaler()
    train_data[selected_columns] = scaler.fit_transform(train_data[selected_columns])
    test_data[selected_columns] = scaler.transform(test_data[selected_columns])
    return train_data, test_data

def save_to_pickle(data, filename):
    """
    Save data to a pickle file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(filename):
    """
    Load data from a pickle file.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Main workflow
if __name__ == "__main__":
    # Load training data
    train = pd.concat([read_data(file[0]) for file in TRAIN_DATA_FILES], ignore_index=True)
    train['dataset'] = pd.concat([pd.Series(file[1], index=range(len(read_data(file[0])))) for file in TRAIN_DATA_FILES], ignore_index=True)

    # Compute RUL for training data
    train['max_cycle'] = train.groupby('unit')['time_cycles'].transform('max')
    train['RUL'] = train['max_cycle'] - train['time_cycles']

    # Filter specific dataset
    dataset = train[train['dataset'] == 'FD001']

    # Split the dataset into train and test sets by units
    tr, te = split_units_by_subset(dataset, subset_column='dataset', test_size=0.2)

    # Select relevant sensors and metadata
    tr = tr[SELECTED_SENSORS + ['unit', 'dataset', 'time_cycles']]
    te = te[SELECTED_SENSORS + ['unit', 'dataset', 'time_cycles']]

    # Scale the sensor data
    tr, te = scale_data(tr, te, SELECTED_SENSORS)

    # Save processed datasets to pickle files
    save_to_pickle(tr, './prepared_data/train_data.pkl')
    save_to_pickle(te, './prepared_data/test_data.pkl')

    # Display a confirmation message
    print("Training and testing datasets have been saved to 'train_data.pkl' and 'test_data.pkl'.")
