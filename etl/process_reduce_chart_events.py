import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# relevant ITEMIDs
gcs_eye_opening = [184, 220739, 226756, 227011]
gcs_verbal_response = [723, 223900, 226758, 227014]
gcs_motor_response = [454, 223901, 226757, 227012]
gcs_total = [198, 226755]
diastolic_blood_pressure = [8364, 8368, 8440, 8441, 8502, 8503, 8506, 8555, 220051, 220180, 224643, 225310, 227242]
systolic_blood_pressure = [6, 51, 442, 455, 3313, 3315, 3321, 6701, 220050, 220179, 224167, 225309, 227243]
mean_blood_pressure = [52, 443, 456, 2293, 2294, 2647, 3312, 3314, 3320, 6590, 6702, 6927, 7620, 220052, 220181, 225312]
heart_rate = [211, 220045, 227018]
fraction_inspired_oxygen = [189, 190, 727, 1040, 1206, 1863, 2518, 2981, 3420, 3422, 7018, 7041, 7570, 223835, 226754, 227009, 227010]
respiratory_rate = [614, 615, 618, 619, 651, 653, 1884, 3603, 6749, 7884, 8113, 220210, 224422, 224688, 224689, 224690, 226774, 227050]
body_temperature = [676, 677, 678, 679, 3652, 3654, 6643, 223761, 223762, 226778, 227054]
weight = [763, 3580, 3581, 3582, 3693, 224639, 226512, 226531]
height = [1394, 226707, 226730]

body_temperature_F = [678, 679, 3652, 3654, 6643, 223761, 226778, 227054]
weight_lb = [3581, 226531]
weight_oz = [3582]
height_inch = [1394, 226707]

relevant_ids = gcs_eye_opening + gcs_verbal_response + gcs_motor_response + gcs_total + mean_blood_pressure + heart_rate + fraction_inspired_oxygen + respiratory_rate + body_temperature + weight + height

def inch_to_cm(value):
    return value*2.54

def lb_to_kg(value):
    return value/2.205

def oz_to_kg(value):
    return value/35.274

def f_to_c(value):
    return (value-32)*5/9

def frac_to_perc(value):
    return value*100


def process_reduce_chart_events(data_dir: str, output_dir: str):
    dtype = {
        'SUBJECT_ID': 'int32',
        'HADM_ID': 'int32',
        'ICUSTAY_ID': 'str',
        'ITEMID': 'int32',
        'CHARTTIME': 'str',
        'VALUENUM': 'float32'
    }
    parse_dates = ['CHARTTIME']
    chunksize = 1000000
    first = True

    df: pd.DataFrame
    for df in tqdm(pd.read_csv(os.path.join(data_dir, 'CHARTEVENTS.csv'), usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates, chunksize=chunksize)):
        df = df[df['ICUSTAY_ID'].notna() & df['VALUENUM'].notna() & (df['ITEMID'].isin(relevant_ids)) & (df['VALUENUM'] > 0)]

        # convert units
        df.loc[df['ITEMID'].isin(body_temperature_F), 'VALUENUM'] = f_to_c(df[df['ITEMID'].isin(body_temperature_F)].VALUENUM)
        df.loc[df['ITEMID'].isin(weight_lb), 'VALUENUM'] = lb_to_kg(df[df['ITEMID'].isin(weight_lb)].VALUENUM)
        df.loc[df['ITEMID'].isin(weight_oz), 'VALUENUM'] = oz_to_kg(df[df['ITEMID'].isin(weight_oz)].VALUENUM)
        df.loc[df['ITEMID'].isin(height_inch), 'VALUENUM'] = inch_to_cm(df[df['ITEMID'].isin(height_inch)].VALUENUM)
        df.loc[(df['ITEMID'].isin(fraction_inspired_oxygen)) & (df['VALUENUM']<=1), 'VALUENUM'] = frac_to_perc(df[(df['ITEMID'].isin(fraction_inspired_oxygen)) & (df['VALUENUM']<=1)].VALUENUM)

        # remove implausible measurements
        df = df[~(df['ITEMID'].isin(gcs_total) & (df.VALUENUM < 3))]
        df = df[~(df['ITEMID'].isin(diastolic_blood_pressure + systolic_blood_pressure + mean_blood_pressure) & (df.VALUENUM > 250))]
        df = df[~(df['ITEMID'].isin(heart_rate) & ((df.VALUENUM < 1) | (df.VALUENUM > 250)))]
        df = df[~(df['ITEMID'].isin(fraction_inspired_oxygen) & (df.VALUENUM > 100))]
        df = df[~(df['ITEMID'].isin(respiratory_rate) & ((df.VALUENUM < 1) | (df.VALUENUM > 100)))]
        df = df[~(df['ITEMID'].isin(body_temperature) & (df.VALUENUM > 50))]
        df = df[~(df['ITEMID'].isin(weight) & (df.VALUENUM > 700))]
        df = df[~(df['ITEMID'].isin(height) & (df.VALUENUM > 300))]
        df = df[df['VALUENUM'] > 0]

        # label
        df['CE_TYPE'] = ''
        df.loc[df['ITEMID'].isin(gcs_eye_opening), 'CE_TYPE'] = 'GCS_EYE_OPENING'
        df.loc[df['ITEMID'].isin(gcs_verbal_response), 'CE_TYPE'] = 'GCS_VERBAL_RESPONSE'
        df.loc[df['ITEMID'].isin(gcs_motor_response), 'CE_TYPE'] = 'GCS_MOTOR_RESPONSE'
        df.loc[df['ITEMID'].isin(gcs_total), 'CE_TYPE'] = 'GCS_TOTAL'
        df.loc[df['ITEMID'].isin(diastolic_blood_pressure), 'CE_TYPE'] = 'DIASTOLIC_BP'
        df.loc[df['ITEMID'].isin(systolic_blood_pressure), 'CE_TYPE'] = 'SYSTOLIC_BP'
        df.loc[df['ITEMID'].isin(mean_blood_pressure), 'CE_TYPE'] = 'MEAN_BP'
        df.loc[df['ITEMID'].isin(heart_rate), 'CE_TYPE'] = 'HEART_RATE'
        df.loc[df['ITEMID'].isin(fraction_inspired_oxygen), 'CE_TYPE'] = 'FRACTION_INSPIRED_OXYGEN'
        df.loc[df['ITEMID'].isin(respiratory_rate), 'CE_TYPE'] = 'RESPIRATORY_RATE'
        df.loc[df['ITEMID'].isin(body_temperature), 'CE_TYPE'] = 'BODY_TEMPERATURE'
        df.loc[df['ITEMID'].isin(weight), 'CE_TYPE'] = 'WEIGHT'
        df.loc[df['ITEMID'].isin(height), 'CE_TYPE'] = 'HEIGHT'
        df = df.drop(columns=['ITEMID'])

        if not first:
            df.to_csv(os.path.join(output_dir, 'chart_events_reduced.csv'), index=False, header=False, mode='a')
        else:
            df.to_csv(os.path.join(output_dir, 'chart_events_reduced.csv'), index=False)
            first = False


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

    # takes roughly 5 min (331it) and will produce a 2.03GB CSV
    process_reduce_chart_events(data_dir, output_dir)

