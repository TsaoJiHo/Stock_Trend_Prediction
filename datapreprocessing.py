from tqdm import tqdm
import pandas as pd
import pathlib

def _insert_row_(row_number, df):
    # Slice the upper half of the dataframe
    df1 = df[0:row_number]
  
    # Store the result of lower half of the dataframe
    df2 = df[row_number:]
  
    # Insert the row in the upper half dataframe
    df1.loc[row_number]=df1.iloc[row_number-1]
  
    # Concat the two dataframes
    df_result = pd.concat([df1, df2])
  
    # Reassign the index labels
    df_result.index = [*range(df_result.shape[0])]
  
    # Return the updated dataframe
    return df_result

def preprocess(path, label):
    label_loc_map = {'10d': 25, '20d': 26, '60d': 27, '120d': 28}
    df = pd.read_csv(path)
    df = df.iloc[:, list(range(25)) + [label_loc_map[label]]]
    df = df.dropna()
    
    # split test data
    
    testing_df = df[df['date'] >= '2021-01-01']
    testing_date = pd.read_csv(f'./testing_date_{label}.csv')['date']
    for j, date in enumerate(testing_date):
        if date not in testing_df['date'].values:
            testing_df = _insert_row_(j, testing_df)
    testing_df.to_csv(pathlib.Path(path.parent/(f'testing_data_{path.stem.split("_")[-1]}_{label}')).with_suffix('.csv'))

    # oversampling training data
    df = df[df['date'] < '2021-01-01']
    value_counts_list = [df[label].value_counts()[0], df[label].value_counts()[1], df[label].value_counts()[2]]

    down_dif = max(value_counts_list) - value_counts_list[0]
    stable_dif = max(value_counts_list) - value_counts_list[1]
    up_dif = max(value_counts_list) - value_counts_list[2]


    resample_df = [df, df[df[label] == 0].sample(n = down_dif, replace=(value_counts_list[0] < down_dif))
    , df[df[label] == 1].sample(n = stable_dif, replace=(value_counts_list[1] < stable_dif))
    , df[df[label] == 2].sample(n = up_dif, replace=(value_counts_list[2] < up_dif))]


    pd.concat(resample_df).to_csv(pathlib.Path(path.parent/(f'training_data_{path.stem.split("_")[-1]}_{label}')).with_suffix('.csv'))



if __name__ == '__main__':

    for path in tqdm(pathlib.Path('./data/training_data').glob('*.csv')):
        for label in ['10d', '20d', '60d', '120d']:
            
            preprocess(path, label)

