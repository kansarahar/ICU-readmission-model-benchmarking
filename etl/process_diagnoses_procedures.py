import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_admissions(data_dir: str):
    print('loading admissions table')
    dtype = {
        'HADM_ID': 'int32',
        'ADMITTIME': 'str',
        'DISCHTIME': 'str'
    }
    parse_dates = ['ADMITTIME', 'DISCHTIME']
    admissions = pd.read_csv(os.path.join(data_dir, 'ADMISSIONS.csv'), usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
    print('loaded admissions table')
    return admissions

def load_patient_admissions(output_dir: str):
    print('loading patient_admissions table from .pkl')
    patient_admissions = pd.read_pickle(os.path.join(output_dir, 'patient_admissions.pkl'))
    print('loaded patient_admissions table')
    return patient_admissions

def load_diagnoses_and_procedures(data_dir: str, admissions: pd.DataFrame, patient_admissions: pd.DataFrame):
    print('loading diagnoses and procedures table')
    dtype = {
        'SUBJECT_ID': 'int32',
        'HADM_ID': 'int32',
        'ICD9_CODE': 'str'
    }

    # contains ICD diagnoses data
    diagnoses = pd.read_csv(os.path.join(data_dir, 'DIAGNOSES_ICD.csv'), usecols=dtype.keys(), dtype=dtype)
    diagnoses = diagnoses.dropna()

    # contains ICD procedure data
    procedures = pd.read_csv(os.path.join(data_dir, 'PROCEDURES_ICD.csv'), usecols=dtype.keys(), dtype=dtype)
    procedures = procedures.dropna()

    # merge diagnoses and procedures
    diagnoses['ICD9_CODE'] = 'DIAGN_' + diagnoses['ICD9_CODE'].str.lower().str.strip()
    procedures['ICD9_CODE'] = 'PROCE_' + procedures['ICD9_CODE'].str.lower().str.strip()
    diag_proc = pd.concat([diagnoses, procedures], ignore_index=True, sort=False)

    # link diagnoses/procedures and admissions
    diag_proc = pd.merge(diag_proc, admissions, how='inner', on='HADM_ID').drop(columns=['HADM_ID'])
    # link diagnoses/procedures and patient_admissions
    diag_proc = pd.merge(patient_admissions[['SUBJECT_ID', 'ICUSTAY_ID', 'OUTTIME']], diag_proc, how='left', on=['SUBJECT_ID'])

    # remove codes related to future admissions
    diag_proc['DAYS_TO_OUT'] = (diag_proc['OUTTIME'] - diag_proc['ADMITTIME']) / np.timedelta64(1, 'D')
    diag_proc = diag_proc[(diag_proc['DAYS_TO_OUT'] >= 0) | diag_proc['DAYS_TO_OUT'].isna()]

    # reset time value
    diag_proc['DAYS_TO_OUT'] = (diag_proc['OUTTIME'] - diag_proc['DISCHTIME']) / np.timedelta64(1, 'D')
    diag_proc.loc[diag_proc['DAYS_TO_OUT'] < 0, 'DAYS_TO_OUT'] = 0
    diag_proc = diag_proc.drop(columns=['SUBJECT_ID', 'OUTTIME', 'ADMITTIME', 'DISCHTIME'])

    # not sure why they added the line below as it's causing issues - commenting it out
    # diag_proc = pd.merge(patient_admissions[['ICUSTAY_ID']], diag_proc, how='left', on='ICUSTAY_ID')
    diag_proc = diag_proc.drop_duplicates()

    # lump all infrequent ICD9 codes together
    diag_proc = diag_proc.apply(lambda col: col.mask(col.map(col.value_counts() < 100), 'other') if col.name in ['ICD9_CODE'] else col)

    diag_proc = diag_proc.sort_values(by=['ICUSTAY_ID', 'DAYS_TO_OUT'], ascending=[True, True])
    print('loading diagnoses and procedures table')
    return diag_proc

def process_diagnoses_procedures(data_dir: str, output_dir: str):
    admissions = load_admissions(data_dir)
    patient_admissions = load_patient_admissions(output_dir)
    diag_proc = load_diagnoses_and_procedures(data_dir, admissions, patient_admissions)

    print('Saving Diagnoses and Procedures table...')
    os.makedirs(output_dir, exist_ok=True)
    diag_proc.to_pickle(os.path.join(output_dir, 'diag_proc.pkl'))
    diag_proc.to_csv(os.path.join(output_dir, 'diag_proc.csv'), index=False)
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

    process_diagnoses_procedures(data_dir, output_dir)

