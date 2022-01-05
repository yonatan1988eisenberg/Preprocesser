import os
import logging
import warnings
import argparse
import datetime
import numpy as np
import pandas as pd
import recordlinkage as rl
from collections import Counter
from configparser import ConfigParser
from recordlinkage.preprocessing import clean

warnings.simplefilter(action='ignore', category=FutureWarning)

# read config file
config_object = ConfigParser()
config_object.read("pr_config.ini")


def write_multiple_dfs(writer, df_list, sheets, logger, spaces=1):
    """
    This function writes the multiple dataframes in df_list to the excel sheet in filename
    """

    row = 0
    for dataframe in df_list:
        dataframe.to_excel(writer, sheet_name=sheets, startrow=row, startcol=0)
        row = row + len(dataframe.index) + spaces + 1

    logger.info('wrote a sheet to the output file')


def get_shared_pairs(top_features, df, ind_list):
    """
    this function gets a table with two columns ['ind1', 'ind2] and looks for new rows which were compared
    with at least 2 element from ind_list. If the shared row was handled it will be excluded.
    example:
    table:
    ind1  ind2
    x     y
    x     z
    x     a
    z     y

    ind_list:
    [x,y]

    should return [z] since it is compared to both x and y. a is excluded since it is only compared to x
    """

    pairs_list = [get_my_pairs(i, top_features) for i in ind_list]
    counter = Counter(x for xs in pairs_list for x in set(xs))
    for i, val in counter.most_common():
        if len(ind_list) >= int(config_object['const']['max_items_per_cluster']):
            break
        if (val > 1) and (i not in ind_list) and (df.loc[i]['handled'] == 0):
            ind_list.append(i)

    return ind_list


def get_my_pairs(main_ind, top_features):
    """
    This function gets a main index and a table that has the columns ['ind1', 'ind2']
    it returns a list of all the indices (inc. main_ind) which appear in those columns where
     main_ind also appears
    """
    indices_with_main = top_features[(top_features['ind1'] == main_ind) |
                                     (top_features['ind2'] == main_ind)].index
    return list(np.unique(top_features.loc[indices_with_main, ['ind1', 'ind2']].values))


def most_common_item(top_features, df):
    """
    this fucntion gets a table with two columns ['ind1', 'ind2'] and returns the most common value in them
    """
    df1 = top_features.groupby('ind1').size().sort_values()
    df2 = top_features.groupby('ind2').size().sort_values()
    freqs = pd.concat([df1, df2]).groupby(level=[0]).sum().reset_index().rename(
        columns={0: 'freq', 'index': 'row'}).sort_values(by='freq',
                                                         ascending=False).reset_index()[['row', 'freq']]

    # skip handled rows
    flag = True
    c = 0
    while flag and freqs.shape[0] > c:
        if df.loc[freqs.iloc[c].row, 'handled'] == 0:
            flag = False
        c += 1

    return freqs.iloc[c - 1].row


def remove_handled(features, df):
    """
    this function removes rows from table where both the values in the columns ['ind1', 'ind2']
    are marked as handled in the df
    """
    handled_inds = df[df.handled == 1].index.to_list()
    if handled_inds:
        drop_inds = features[(features['ind1'].isin(handled_inds)) | (features['ind2'].isin(handled_inds))].index
        features.drop(index=drop_inds, inplace=True)


def find_similar_items(raw_data, df, features, logger):
    """
    This function finds similar items in df by its similarity score which can be found in features
    """
    top = int(config_object['const']['top'])
    display_columns = ['name', 'address', 'about', 'tags']
    df['handled'] = 0
    features = features.copy().reset_index().rename(columns={'level_0': 'ind1', 'level_1': 'ind2'})
    clusters_df_list = []

    while 0 in df['handled'].value_counts().index:
        # sort features
        features.sort_values(by='address_similarity', ascending=False, inplace=True)

        # remove the pair where both items where handled
        remove_handled(features, df)

        # get the top rows from features
        top_features = features.iloc[:top]

        # find the most frequent item in top features
        freq_item = most_common_item(top_features, df)

        # and its pairs in top_features
        inds_to_compare = get_my_pairs(freq_item, top_features)

        # add shared pairs
        final_items_list = get_shared_pairs(top_features, df, inds_to_compare)

        # add the cluster to the list
        clusters_df_list.append(raw_data.loc[final_items_list][display_columns])

        # mark the items in the cluster as 'handled'
        df.loc[final_items_list, 'handled'] = 1

    logger.info('found similar clusters')
    return clusters_df_list


def extract_features(df, logger):
    """
    This function extracts the features, using the RecordLinkage package
    """

    # index
    indexer = rl.Index()
    indexer.add(rl.index.Full())
    pairs = indexer.index(df)

    # lower case the address column
    df[config_object['const']['compare_col']] = clean(df[config_object['const']['compare_col']])

    # compare and compute
    compare = rl.Compare()
    compare.string(config_object['const']['compare_col'], config_object['const']['compare_col'],
                   'jarowinkler', label='address_similarity')

    features = compare.compute(pairs, df)
    logger.info('extracted the features')

    return features


def sort_and_remove_duplicates(raw_data, logger):
    """
    This function sorts the dataframe by the column 'number_of_reviews' and removes
    duplicates by the column 'name'
    """
    df = raw_data.copy()

    try:
        df.sort_values(config_object['const']['sort_col'], ascending=False, inplace=True)
        df.drop_duplicates(subset=config_object['const']['duplicates_col'], inplace=True)
    except KeyError as er:
        logger.debug('sort_col or duplicates_col is missing, check the log and/or config file for more details')
        print(er)
        exit(1)

    logger.info('sorted and removed duplicates')
    return df


def val_input(args, logger):
    """
    This function validated that the input file exists and that the output path's folder exists
    """

    if not os.path.isfile(args.path):
        logger.debug('the input file doesn\'t exists')
        return False

    if args.save:
        if '/' in args.save:
            folder = "/".join(args.save.split('/')[:-1])
            if not os.path.exists(folder):
                logger.debug('the output folder doesn\'t exists')
                return False
        else:
            folder = "/".join(args.path.split('/')[:-1])
            args.save = folder + '/' + args.save

    else:
        current_time = datetime.datetime.now()
        save_path = f'processed_data_{current_time.year}-{current_time.month}-{current_time.day}-{current_time.hour}-' \
                    f'{current_time.minute}.xlsx'
        args.save = save_path
        logger.info('the save path was set to default')
    logger.info(f'args={args}')
    logger.info('input was validated')
    return True


def parse_args(logger):
    """
    This function initialize the parser and the input parameters
    """

    my_parser = argparse.ArgumentParser(description=config_object['params']['description'])
    my_parser.add_argument('--path', '-p',
                           required=True,
                           type=str,
                           help=config_object['params']['path_help'])

    my_parser.add_argument('--save', '-s',
                           required=False,
                           type=str, default=None,
                           help=config_object['params']['save_help'])

    args = my_parser.parse_args()
    logger.info('Parsed arguments')
    return args


def init_logger():
    """
    This function initialize the logger and returns its handle
    """

    log_formatter = logging.Formatter('%(levelname)s-%(asctime)s-FUNC:%(funcName)s-LINE:%(lineno)d-%(message)s')
    logger = logging.getLogger('log')
    logger.setLevel('DEBUG')
    file_handler = logging.FileHandler('pr_log.txt')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    return logger


def main():

    # config logger
    logger = init_logger()
    logger.info('STARTED RUNNING')

    # parse args
    args = parse_args(logger)

    # validate the input
    if not val_input(args, logger):
        logger.debug('terminating due to invalid input')
        exit()

    # load the data
    raw_data = pd.read_csv(args.path, encoding='UTF-8')

    # sort by number of reviews and remove duplicates by name
    reduced_data = sort_and_remove_duplicates(raw_data, logger)

    # extract features
    features = extract_features(reduced_data, logger)

    # find similar items
    similar_list = find_similar_items(raw_data, reduced_data, features, logger)

    # set handled = 0
    reduced_data['handled'] = 0

    # save the reduced data and clusters to excel file
    logger.info(f'save location: {args.save}')
    with pd.ExcelWriter(args.save, engine='openpyxl') as writer:
        write_multiple_dfs(writer, [reduced_data], 'Data', logger)
        write_multiple_dfs(writer, similar_list, 'Near_by', logger)

    print(f'Done! the file can fe found at {args.save}')
    logger.info(f'Done! the file can fe found at {args.save}')


if __name__ == '__main__':
    main()
