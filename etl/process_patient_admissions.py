import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_icu_stays_table(data_dir: str):
    print('loading ICU Stays table...')
    dtype = {
        'SUBJECT_ID': 'int32',
        'HADM_ID': 'int32',
        'ICUSTAY_ID': 'int32',
        'INTIME': 'str',
        'OUTTIME': 'str',
        'LOS': 'float32'
    }
    parse_dates = ['INTIME', 'OUTTIME']
    icu_stays = pd.read_csv(os.path.join(data_dir, 'ICUSTAYS.csv'), usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
    print('loaded ICU Stays table')
    return icu_stays

def load_patients_table(data_dir: str):
    print('loading Patients table')
    dtype = {
        'SUBJECT_ID': 'int32',
        'GENDER': 'str',
        'DOB': 'str',
        'DOD': 'str'
    }
    parse_dates = ['DOB', 'DOD']
    patients = pd.read_csv(os.path.join(data_dir, 'PATIENTS.csv'), usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)

    # honestly not sure why they shifted these patients DOB by this specific amount...
    # in fact it's kinda weird because some people have birthdays that haven't happened yet
    patients.loc[patients['DOB'].dt.year < 2000, 'DOB'] = patients['DOB'] + pd.DateOffset(years=(300-91), days=(-0.4*365))
    print('loaded Patients table')

    return patients

def load_admissions_table(data_dir: str):
    print('loading Admissions table')
    dtype = {
        'SUBJECT_ID': 'int32',
        'HADM_ID': 'int32',
        'ADMISSION_LOCATION': 'str',
        'INSURANCE': 'str',
        'MARITAL_STATUS': 'str',
        'ETHNICITY': 'str',
        'ADMITTIME': 'str',
        'ADMISSION_TYPE': 'str'
    }
    parse_dates = ['ADMITTIME']
    admissions = pd.read_csv(os.path.join(data_dir, 'ADMISSIONS.csv'), usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
    print('loaded Admissions table')
    return admissions

def load_services_table(data_dir: str):
    print('loading Services table')
    dtype = {
        'SUBJECT_ID': 'int32',
        'HADM_ID': 'int32',
        'TRANSFERTIME': 'str',
        'CURR_SERVICE': 'str'
    }
    parse_dates = ['TRANSFERTIME']
    services = pd.read_csv(os.path.join(data_dir, 'SERVICES.csv'), usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
    print('loaded Services table')
    return services

def link_icu_patient_tables(icu_stays: pd.DataFrame, patients: pd.DataFrame):
    print('linking icu_stays and patients tables')
    icu_patient_table = pd.merge(icu_stays, patients, how='inner', on=['SUBJECT_ID'])
    icu_patient_table = icu_patient_table.sort_values(by=['SUBJECT_ID', 'OUTTIME'], ascending=[True, False])
    assert len(icu_patient_table['SUBJECT_ID'].unique()) == 46476
    assert len(icu_patient_table['ICUSTAY_ID'].unique()) == 61532

    # exclude ICU stays where the patient dies
    icu_patient_table = icu_patient_table[~(icu_patient_table['DOD'] <= icu_patient_table['OUTTIME'])]
    assert len(icu_patient_table['SUBJECT_ID'].unique()) == 43126
    assert len(icu_patient_table['ICUSTAY_ID'].unique()) == 56745

    # determine number of icu discharges in the last 365 days
    icu_patient_table['NUM_RECENT_ADMISSIONS'] = 0
    for name, group in tqdm(icu_patient_table.groupby(['SUBJECT_ID'])):
        for index, row in group.iterrows():
            days_diff = (row['OUTTIME'] - group['OUTTIME']).dt.days
            icu_patient_table.at[index, 'NUM_RECENT_ADMISSIONS'] = len(group[(days_diff > 0) & (days_diff <= 365)])

    # create age variable and exclude patients under 18 years old
    icu_patient_table['AGE'] = (icu_patient_table['OUTTIME'] - icu_patient_table['DOB']).dt.days/365.0
    icu_patient_table = icu_patient_table[icu_patient_table['AGE'] >= 18]
    assert len(icu_patient_table['SUBJECT_ID'].unique()) == 35233
    assert len(icu_patient_table['ICUSTAY_ID'].unique()) == 48616

    # calculate time difference between dischange and readmission
    icu_patient_table['DAYS_TO_NEXT'] = (icu_patient_table.groupby('SUBJECT_ID').shift(1)['INTIME'] - icu_patient_table['OUTTIME']).dt.days

    # add early readmission flag for readmission within 30 days
    icu_patient_table['POSITIVE'] = icu_patient_table['DAYS_TO_NEXT'] <= 30
    assert icu_patient_table['POSITIVE'].sum() == 5495

    # add early death flag for death within 30 days of admission
    early_death = (icu_patient_table['DOD'] - icu_patient_table['OUTTIME']).dt.days <= 30
    assert early_death.sum() == 3795

    # remove patients who died within less than 30 days after being discharged (no chance of readmission)
    icu_patient_table = icu_patient_table[icu_patient_table['POSITIVE'] | ~early_death]
    assert len(icu_patient_table['SUBJECT_ID'].unique()) == 33150
    assert len(icu_patient_table['ICUSTAY_ID'].unique()) == 45298

    icu_patient_table = icu_patient_table.drop(columns=['DOB', 'DOD', 'DAYS_TO_NEXT'])
    print('linked icu_stays and patients tables')
    return icu_patient_table

def link_icu_patient_admission_tables(icu_patient_table: pd.DataFrame, admissions_table: pd.DataFrame):
    print('linking with admissions table')
    icu_patient_admission_table = pd.merge(icu_patient_table, admissions_table, how='left', on=['SUBJECT_ID', 'HADM_ID'])

    icu_patient_admission_table.loc[icu_patient_admission_table['ETHNICITY'].str.contains('WHITE'), 'ETHNICITY'] = 'WHITE'
    icu_patient_admission_table.loc[icu_patient_admission_table['ETHNICITY'].str.contains('BLACK'), 'ETHNICITY'] = 'BLACK/AFRICAN AMERICAN'
    icu_patient_admission_table.loc[icu_patient_admission_table['ETHNICITY'].str.contains('ASIAN'), 'ETHNICITY'] = 'ASIAN'
    icu_patient_admission_table.loc[icu_patient_admission_table['ETHNICITY'].str.contains('HISPANIC'), 'ETHNICITY'] = 'HISPANIC/LATINO'
    icu_patient_admission_table.loc[icu_patient_admission_table['ETHNICITY'].str.contains('DECLINED'), 'ETHNICITY'] = 'OTHER/UNKNOWN'
    icu_patient_admission_table.loc[icu_patient_admission_table['ETHNICITY'].str.contains('MULTI'), 'ETHNICITY'] = 'OTHER/UNKNOWN'
    icu_patient_admission_table.loc[icu_patient_admission_table['ETHNICITY'].str.contains('UNKNOWN'), 'ETHNICITY'] = 'OTHER/UNKNOWN'
    icu_patient_admission_table.loc[icu_patient_admission_table['ETHNICITY'].str.contains('OTHER'), 'ETHNICITY'] = 'OTHER/UNKNOWN'

    icu_patient_admission_table['MARITAL_STATUS'].fillna('UNKNOWN', inplace=True)
    icu_patient_admission_table.loc[icu_patient_admission_table['MARITAL_STATUS'].str.contains('MARRIED'), 'MARITAL_STATUS'] = 'MARRIED/LIFE PARTNER'
    icu_patient_admission_table.loc[icu_patient_admission_table['MARITAL_STATUS'].str.contains('LIFE PARTNER'), 'MARITAL_STATUS'] = 'MARRIED/LIFE PARTNER'
    icu_patient_admission_table.loc[icu_patient_admission_table['MARITAL_STATUS'].str.contains('WIDOWED'), 'MARITAL_STATUS'] = 'WIDOWED/DIVORCED/SEPARATED'
    icu_patient_admission_table.loc[icu_patient_admission_table['MARITAL_STATUS'].str.contains('DIVORCED'), 'MARITAL_STATUS'] = 'WIDOWED/DIVORCED/SEPARATED'
    icu_patient_admission_table.loc[icu_patient_admission_table['MARITAL_STATUS'].str.contains('SEPARATED'), 'MARITAL_STATUS'] = 'WIDOWED/DIVORCED/SEPARATED'
    icu_patient_admission_table.loc[icu_patient_admission_table['MARITAL_STATUS'].str.contains('UNKNOWN'), 'MARITAL_STATUS'] = 'OTHER/UNKNOWN'

    # if values in these columns appear less than 100 times, then mark these values as 'OTHER/UNKNOWN'
    columns_to_mask = ['ADMISSION_LOCATION', 'INSURANCE', 'MARITAL_STATUS', 'ETHNICITY']
    icu_patient_admission_table = icu_patient_admission_table.apply(lambda col: col.mask(col.map(col.value_counts()) < 100, 'OTHER/UNKNOWN') if col.name in columns_to_mask else col)
    icu_patient_admission_table = icu_patient_admission_table.apply(lambda col: col.str.title() if col.name in columns_to_mask else col)

    # compute pre-ICU length stay in fractional days
    icu_patient_admission_table['PRE_ICU_LOS'] = (icu_patient_admission_table['INTIME'] - icu_patient_admission_table['ADMITTIME']) / np.timedelta64(1, 'D')
    icu_patient_admission_table.loc[icu_patient_admission_table['PRE_ICU_LOS'] < 0, 'PRE_ICU_LOS'] = 0

    icu_patient_admission_table = icu_patient_admission_table.drop(columns=['ADMITTIME'])
    print('linked with admissions table')
    return icu_patient_admission_table

def link_icu_patient_admission_service_tables(icu_patient_admission_table: pd.DataFrame, services_table: pd.DataFrame):
    print('linking with services table')

    # keep only the first service
    services = services_table.sort_values(by=['HADM_ID', 'TRANSFERTIME'], ascending=True)
    services = services.groupby(['HADM_ID']).nth(0).reset_index()

    # check if first service is a surgery
    services['SURGERY'] = services['CURR_SERVICE'].str.contains('SURG') | services['CURR_SERVICE'] == 'ORTHO'

    icu_patient_admission_service_table = pd.merge(icu_patient_admission_table, services, how='left', on=['SUBJECT_ID', 'HADM_ID'])

    # get elective surgery admissions
    icu_patient_admission_service_table['ELECTIVE_SURGERY'] = ((icu_patient_admission_service_table['ADMISSION_TYPE'] == 'ELECTIVE') & icu_patient_admission_service_table['SURGERY']).astype(int)
    icu_patient_admission_service_table = icu_patient_admission_service_table.sort_values(by='ICUSTAY_ID', ascending=True)

    icu_patient_admission_service_table = icu_patient_admission_service_table.drop(columns=['TRANSFERTIME', 'CURR_SERVICE', 'ADMISSION_TYPE', 'SURGERY'])
    print('linked with services table')
    return icu_patient_admission_service_table

def process_patient_admissions(data_dir: str, output_dir: str):
    icu_stays = load_icu_stays_table(data_dir)
    patients = load_patients_table(data_dir)
    admissions = load_admissions_table(data_dir)
    services = load_services_table(data_dir)

    icu_patient_table = link_icu_patient_tables(icu_stays, patients)
    icu_patient_admission_table = link_icu_patient_admission_tables(icu_patient_table, admissions)
    icu_patient_admission_service_table = link_icu_patient_admission_service_tables(icu_patient_admission_table, services)

    assert len(icu_patient_admission_service_table) == 45298
    print('Saving Patient Admissions table...')
    os.makedirs(output_dir, exist_ok=True)
    icu_patient_admission_service_table.to_pickle(os.path.join(output_dir, 'patient_admissions.pkl'))
    icu_patient_admission_service_table.to_csv(os.path.join(output_dir, 'patient_admissions.csv'), index=False)
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

    process_patient_admissions(data_dir, output_dir)
