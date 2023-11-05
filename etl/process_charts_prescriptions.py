import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_tables(data_dir: str, output_dir: str):
    print('loading tables')

    patient_admissions = pd.read_pickle(os.path.join(output_dir, 'patient_admissions.pkl'))
    chart_outputs = pd.read_pickle(os.path.join(output_dir, 'charts_outputs_reduced.pkl'))

    dtype = {
        'ICUSTAY_ID': 'str',
        'DRUG': 'str',
        'STARTDATE': 'str'
    }
    parse_dates = ['STARTDATE']

    # load prescriptions table
    prescriptions = pd.read_csv(os.path.join(data_dir, 'PRESCRIPTIONS.csv'), usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
    prescriptions = prescriptions.dropna()
    prescriptions['ICUSTAY_ID'] = prescriptions['ICUSTAY_ID'].astype('int32')
    prescriptions['DRUG'] = 'PRESC_' + prescriptions['DRUG'].str.lower().replace('\s+', '', regex=True)
    prescriptions = prescriptions.rename(columns={ 'DRUG': 'VALUECAT', 'STARTDATE': 'CHARTTIME' })

    print('loaded tables')
    return patient_admissions, chart_outputs, prescriptions

def link_tables(patient_admissions: pd.DataFrame, chart_outputs: pd.DataFrame, prescriptions: pd.DataFrame):
    df = pd.concat([chart_outputs, prescriptions], ignore_index=True, sort=False)
    df = pd.merge(patient_admissions[['ICUSTAY_ID', 'OUTTIME']], df, how='left', on=['ICUSTAY_ID'])

    df['HOURS_TO_OUT'] = (df['OUTTIME']-df['CHARTTIME']) / np.timedelta64(1, 'h')
    df.loc[df['HOURS_TO_OUT'] < 0, 'HOURS_TO_OUT'] = 0
    df = df.drop(columns=['OUTTIME', 'CHARTTIME'])
    df = df.drop_duplicates()

    # lump together all infrequent codes
    df = df.apply(lambda col: col.mask(col.map(col.value_counts()) < 100, 'other') if col.name in ['VALUECAT'] else col)
    return df

def process_charts_prescriptions(data_dir: str, output_dir: str):
    patient_admissions, chart_outputs, prescriptions = load_tables(data_dir, output_dir)
    df = link_tables(patient_admissions, chart_outputs, prescriptions)
    assert len(df['ICUSTAY_ID'].unique()) == 45298
    df = df.sort_values(by=['ICUSTAY_ID', 'HOURS_TO_OUT'], ascending=[True, True])

    print('Saving Charts + Prescriptions table')
    df.to_pickle(os.path.join(output_dir, 'charts_prescriptions.pkl'))
    df.to_csv(os.path.join(output_dir, 'charts_prescriptions.csv'), index=False)
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

    process_charts_prescriptions(data_dir, output_dir)