from tqdm import tqdm
import pandas as pd
import pathlib

pd.options.mode.chained_assignment = None  # default='warn'

def _insert_row_(row_number, df):
    # Slice the upper half of the dataframe
    df1 = df[0:row_number]
  
    # Store the result of lower half of the dataframe
    df2 = df[row_number:]
  
    # Insert the row in the upper half dataframe
    # df1.loc[row_number]=df1.iloc[row_number-1]
    df1 = df1.append(df1.iloc[row_number-1] , ignore_index =True)
  
    # Concat the two dataframes
    df_result = pd.concat([df1, df2])
  
    # Reassign the index labels
    df_result.index = [*range(df_result.shape[0])]
  
    # Return the updated dataframe
    return df_result

def preprocess(path, label, test_only = False):
    label_loc_map = {'10d': 25, '20d': 26, '30d':27, '40d':28, '60d': 29, '120d': 30}
    df = pd.read_csv(path) # raw data
    split_date = '2013-01-01'

    # generate close df for stable fluctuation calculating
    close_df = df[['date', 'close']]
    close_df = close_df[close_df['date'] >= split_date]
    close_df = close_df.dropna() # drop missing close

    testing_date = pd.read_csv('./data/indicator_data/taiex.csv')
    testing_date = testing_date[testing_date['date'] >= split_date]['date']

    for j, date in enumerate(testing_date):
        if date not in close_df['date'].values:
            close_df = _insert_row_(j, close_df)
    close_df.to_csv(pathlib.Path(path.parent/(f'close_df_{path.stem.split("_")[-1]}')).with_suffix('.csv'))

    df = df.iloc[:, list(range(25)) + [label_loc_map[label]]]
    df = df.dropna() # drop front, rear and missing close

    # split test data
    testing_df = df[df['date'] >= split_date]
    # testing_df = testing_df[testing_df['date'] <= '2011-03-02']
    d = int(label[:-1])

    testing_date = testing_date[:-d]

    for j, date in enumerate(testing_date):
        if date not in testing_df['date'].values:
            testing_df = _insert_row_(j, testing_df)
    testing_df.to_csv(pathlib.Path(path.parent/(f'8y_testing_data_{path.stem.split("_")[-1]}_{label}')).with_suffix('.csv'))

    if test_only:
        return

    # oversampling training data
    df = df[df['date'] < split_date]
    value_counts_list = [df[label].value_counts()[0], df[label].value_counts()[1], df[label].value_counts()[2]]

    down_dif = max(value_counts_list) - value_counts_list[0]
    stable_dif = max(value_counts_list) - value_counts_list[1]
    up_dif = max(value_counts_list) - value_counts_list[2]


    resample_df = [df, df[df[label] == 0].sample(n = down_dif, replace=(value_counts_list[0] < down_dif))
    , df[df[label] == 1].sample(n = stable_dif, replace=(value_counts_list[1] < stable_dif))
    , df[df[label] == 2].sample(n = up_dif, replace=(value_counts_list[2] < up_dif))]


    pd.concat(resample_df).to_csv(pathlib.Path(path.parent/(f'training_data_{path.stem.split("_")[-1]}_{label}')).with_suffix('.csv'))



if __name__ == '__main__':
    error_list = []
    for path in tqdm(pathlib.Path('./data/training_data').glob('raw_data_*.csv')):
        for label in ['10d', '20d', '30d', '40d', '60d', '120d']:

            try:
                preprocess(path, label, test_only=True)
            except Exception as e:
                error_list.append((path, label, e))
                continue
    print(error_list)
