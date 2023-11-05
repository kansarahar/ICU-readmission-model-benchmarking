import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_arrays(df: pd.DataFrame, patient_admissions: pd.DataFrame, num_icu_stays: int, code_column: str, time_column: str, quantile=1.):
    df['COUNT'] = df.groupby(['ICUSTAY_ID']).cumcount()
    df = df[df['COUNT'] < df.groupby(['ICUSTAY_ID']).size().quantile(q=quantile)]
    max_count_df = df['COUNT'].max() + 1
    multiindex_df = pd.MultiIndex.from_product([patient_admissions['ICUSTAY_ID'], range(max_count_df)], names=['ICUSTAY_ID', 'COUNT'])
    df = df.set_index(['ICUSTAY_ID', 'COUNT'])


    df = df.reindex(multiindex_df).fillna(0)
    df_times = df[time_column].values.reshape((num_icu_stays, max_count_df))
    df[code_column] = df[code_column].astype('category')
    dict_df = dict(enumerate(df[code_column].cat.categories))
    df[code_column] = df[code_column].cat.codes
    df = df[code_column].values.reshape((num_icu_stays, max_count_df))

    return df, df_times, dict_df

def create_splits(data_dir: str, output_dir: str):

    patient_admissions: pd.DataFrame = pd.read_pickle(os.path.join(output_dir, 'patient_admissions.pkl'))
    dp: pd.DataFrame = pd.read_pickle(os.path.join(output_dir, 'diag_proc.pkl'))
    cp: pd.DataFrame = pd.read_pickle(os.path.join(output_dir, 'charts_prescriptions.pkl'))
    num_icu_stays = len(patient_admissions['ICUSTAY_ID'])

    patient_admissions = pd.get_dummies(patient_admissions, columns=['ADMISSION_LOCATION', 'INSURANCE', 'MARITAL_STATUS', 'ETHNICITY'])
    patient_admissions = patient_admissions.drop(columns=['ADMISSION_LOCATION_Emergency Room Admit', 'INSURANCE_Medicare', 'MARITAL_STATUS_Married/Life Partner', 'ETHNICITY_White'])
    static_columns = patient_admissions.columns.str.contains('AGE|GENDER_M|LOS|NUM_RECENT_ADMISSIONS|ADMISSION_LOCATION|INSURANCE|MARITAL_STATUS|ETHNICITY|PRE_ICU_LOS|ELECTIVE_SURGERY')
    static = patient_admissions.loc[:, static_columns].values
    static_vars = patient_admissions.loc[:, static_columns].columns.values.tolist()

    # classification labels (readmitted within 30 days or not)
    label = patient_admissions.loc[:, 'POSITIVE'].values

    dp, dp_times, dict_dp = get_arrays(dp, patient_admissions, num_icu_stays, 'ICD9_CODE', 'DAYS_TO_OUT', 1)
    cp, cp_times, dict_cp = get_arrays(cp, patient_admissions, num_icu_stays, 'VALUECAT', 'HOURS_TO_OUT', 0.95)
    # normalize times
    dp_times = dp_times / dp_times.max()
    cp_times = cp_times / cp_times.max()

    patients = patient_admissions['SUBJECT_ID'].drop_duplicates()
    train, validate, test = np.split(patients.sample(frac=1, random_state=123), [int(0.7*len(patients)), int(0.8*len(patients))])
    train_ids = patient_admissions['SUBJECT_ID'].isin(train).values
    validate_ids = patient_admissions['SUBJECT_ID'].isin(validate).values
    test_ids = patient_admissions['SUBJECT_ID'].isin(test).values

    test_ids_patients = patient_admissions['SUBJECT_ID'].iloc[test_ids].reset_index(drop=True)# np.savez(hp.data_dir + 'data_arrays.npz', static=static, static_vars=static_vars, label=label,

    np.savez(os.path.join(output_dir, 'data_arrays.npz'), static=static, static_vars=static_vars, label=label, dp=dp, cp=cp, dp_times=dp_times, cp_times=cp_times, dict_dp=dict_dp, dict_cp=dict_cp, train_ids=train_ids, validate_ids=validate_ids, test_ids=test_ids)
    np.savez(os.path.join(output_dir, 'data_dictionaries.npz'), dict_dp=dict_dp, dict_cp=dict_cp)
    test_ids_patients.to_pickle(os.path.join(output_dir, 'test_ids_patients.pkl'))


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

    create_splits(data_dir, output_dir)