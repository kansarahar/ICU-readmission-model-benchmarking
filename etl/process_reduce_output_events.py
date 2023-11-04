import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


# Relevant ITEMIDs, from https://github.com/vincentmajor/mimicfilters/blob/master/lists/OASIS_components/preprocess_urine_awk_str.txt
urine_output = [42810, 43171, 43173, 43175, 43348, 43355, 43365, 43372, 43373, 43374, 43379, 43380, 43431, 43462, 43522, 40405, 40428, 40534,
40288, 42042, 42068, 42111, 42119, 42209, 41857, 40715, 40056, 40061, 40085, 40094, 40096, 42001, 42676, 42556, 43093, 44325, 44706,
44506, 42859, 44237, 44313, 44752, 44824, 44837, 43576, 43589, 43633, 44911, 44925, 42362, 42463, 42507, 42510, 40055, 40057, 40065,
40069, 45804, 45841, 43811, 43812, 43856, 43897, 43931, 43966, 44080, 44103, 44132, 45304, 46177, 46532, 46578, 46658, 46748, 40651,
43053, 43057, 40473, 42130, 41922, 44253, 44278, 46180, 44684, 43333, 43347, 42592, 42666, 42765, 42892, 45927, 44834, 43638, 43654,
43519, 43537, 42366, 45991, 46727, 46804, 43987, 44051, 227489, 226566, 226627, 226631, 45415, 42111, 41510, 40055, 226559, 40428,
40580, 40612, 40094, 40848, 43685, 42362, 42463, 42510, 46748, 40972, 40973, 46456, 226561, 226567, 226632, 40096, 40651, 226557,
226558, 40715, 226563]

def load_item_definitions(data_dir: str):
    print('loading item definitions')
    dtype = {
        'ITEMID': 'int32',
        'LABEL': 'str',
        'UNITNAME': 'str',
        'LINKSTO': 'str'
    }
    item_definitions = pd.read_csv(os.path.join(data_dir, 'D_ITEMS.csv'), usecols=dtype.keys(), dtype=dtype)

    # only want urine outputs
    item_definitions = item_definitions[item_definitions['ITEMID'].isin(urine_output)]
    item_definitions['LABEL'] = item_definitions['LABEL'].str.lower()

    # remove measurements in /kg/hr/nephro
    item_definitions = item_definitions[~(item_definitions['LABEL'].str.contains('hr') | item_definitions['LABEL'].str.contains('kg')) | item_definitions['LABEL'].str.contains('nephro')]
    print('loaded item definitions')
    return item_definitions

def load_output_events(data_dir: str, urine_output: list):
    print('loading output events')
    dtype = {
        'ICUSTAY_ID': 'str',
        'ITEMID': 'int32',
        'CHARTTIME': 'str',
        'VALUE': 'float32'
    }
    parse_dates = ['CHARTTIME']
    output_events = pd.read_csv(os.path.join(data_dir, 'OUTPUTEVENTS.csv'), usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
    output_events = output_events.rename(columns={ 'VALUE': 'VALUENUM' })
    output_events = output_events[output_events['ICUSTAY_ID'].notna() & output_events['VALUENUM'].notna() & output_events['ITEMID'].isin(urine_output) & output_events['VALUENUM'] > 0]
    output_events['ICUSTAY_ID'] = output_events['ICUSTAY_ID'].astype('int32')

    # remove implausible measurements
    output_events = output_events[~(output_events['VALUENUM'] > 10000)]

    # sum all outputs in one day
    output_events = output_events.drop(columns=['ITEMID'],)
    output_events['CHARTTIME'] = output_events['CHARTTIME'].dt.date
    output_events = output_events.groupby(['ICUSTAY_ID', 'CHARTTIME']).sum()
    output_events['CE_TYPE'] = 'URINE_OUTPUT'
    output_events = output_events[~(output_events['VALUENUM'] > 10000)]

    print('loading output events')
    return output_events


def load_icu_stays(data_dir: str):
    print('loading ICU stays')
    dtype = {
        'ICUSTAY_ID': 'int32',
        'INTIME': 'str',
        'OUTTIME': 'str',
    }
    parse_dates = ['INTIME', 'OUTTIME']
    icu_stays = pd.read_csv(os.path.join(data_dir, 'ICUSTAYS.csv'), usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
    icu_stays['INTIME'] = icu_stays['INTIME'].dt.date
    icu_stays['OUTTIME'] = icu_stays['OUTTIME'].dt.date

    print('loaded ICU stays')
    return icu_stays

def merge_output_icu(output_events: pd.DataFrame, icu_stays: pd.DataFrame):
    tmp = icu_stays[['ICUSTAY_ID', 'INTIME']].drop_duplicates()
    tmp = tmp.rename(columns={ 'INTIME': 'CHARTTIME' })
    tmp['ID_IN'] = 1
    df = pd.merge(output_events, tmp, how='left', on=['ICUSTAY_ID', 'CHARTTIME'])
    tmp = icu_stays[['ICUSTAY_ID', 'OUTTIME']].drop_duplicates()
    tmp = tmp.rename(columns={ 'OUTTIME': 'CHARTTIME' })
    tmp['ID_OUT'] = 1
    df = pd.merge(df, tmp, how='left', on=['ICUSTAY_ID', 'CHARTTIME'])

    # remove admission and discharge days
    df = df[df['ID_IN'].isnull() & df['ID_OUT'].isnull()]
    df = df.drop(columns=['ID_IN', 'ID_OUT'])

    # add SUBJECT_ID and HADM_ID
    icu_stays = icu_stays.drop(columns=['INTIME', 'OUTTIME'])
    df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME']) + pd.DateOffset(hours=12)
    return df

def process_reduce_output_events(data_dir: str, output_dir: str):
    item_definitions = load_item_definitions(data_dir)
    urine_output = item_definitions['ITEMID'].tolist()
    output_events = load_output_events(data_dir, urine_output)
    icu_stays = load_icu_stays(data_dir)
    output_events_reduced = merge_output_icu(output_events, icu_stays)

    print('Saving Reduced Output Events table...')
    output_events_reduced.to_pickle(os.path.join(output_dir, 'output_events_reduced.pkl'))
    output_events_reduced.to_csv(os.path.join(output_dir, 'output_events_reduced.csv'), index=False)
    print('Saved table!')


if __name__ == '__main__':

    dir_name = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Transforms the raw data')
    parser.add_argument('--input_path', '-i', dest='input_path', type=str, default='data/mimic-iii-data', help='relative path to data directory to retrieve csv files (relative to main dir)')
    parser.add_argument('--output_path', '-o', dest='output_path', type=str, default='data/preprocessed', help='relative path to data directory to store pickled files (relative to main dir)')

    args = parser.parse_args()

    data_dir = os.path.abspath(os.path.join(dir_name, '..', args.input_path))
    output_dir = os.path.abspath(os.path.join(dir_name, '..', args.output_path))

    print('Selected input file directory:', data_dir)
    print('Selected output file directory:', output_dir)

    process_reduce_output_events(data_dir, output_dir)