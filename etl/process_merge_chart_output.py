import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_chart_events_table(output_dir: str):
    print('loading chart events')
    dtype = {
        'SUBJECT_ID': 'int32',
        'ICUSTAY_ID': 'int32',
        'CE_TYPE': 'str',
        'CHARTTIME': 'str',
        'VALUENUM': 'float32'
    }
    parse_dates = ['CHARTTIME']
    chart_events = pd.read_csv(os.path.join(output_dir, 'chart_events_reduced.csv'), usecols=dtype.keys(), dtype=dtype, parse_dates=parse_dates)
    chart_events = chart_events.sort_values(by=['SUBJECT_ID', 'ICUSTAY_ID', 'CHARTTIME'], ascending=[True, True, False])
    print('loaded chart events')
    return chart_events

def compute_bmi_gcs(chart_events: pd.DataFrame):
    # compute BMI
    rows_bmi = (chart_events['CE_TYPE']=='WEIGHT') | (chart_events['CE_TYPE']=='HEIGHT')
    charts_bmi = chart_events[rows_bmi]
    charts_bmi = charts_bmi.pivot_table(index=['SUBJECT_ID', 'ICUSTAY_ID', 'CHARTTIME'], columns='CE_TYPE', values='VALUENUM')
    charts_bmi = charts_bmi.rename_axis(None, axis=1).reset_index()
    charts_bmi['HEIGHT'] = charts_bmi.groupby('SUBJECT_ID')['HEIGHT'].ffill()
    charts_bmi['HEIGHT'] = charts_bmi.groupby('SUBJECT_ID')['HEIGHT'].bfill()
    charts_bmi =  charts_bmi[~pd.isnull(charts_bmi).any(axis=1)]
    charts_bmi['VALUENUM'] = charts_bmi['WEIGHT']/charts_bmi['HEIGHT']/charts_bmi['HEIGHT']*10000
    charts_bmi['CE_TYPE'] = 'BMI'
    charts_bmi = charts_bmi.drop(columns=['HEIGHT', 'WEIGHT'])

    # compute GCS total if not available
    rows_gcs = (chart_events['CE_TYPE']=='GCS_EYE_OPENING') | (chart_events['CE_TYPE']=='GCS_VERBAL_RESPONSE') | (chart_events['CE_TYPE']=='GCS_MOTOR_RESPONSE') | (chart_events['CE_TYPE']=='GCS_TOTAL')
    charts_gcs = chart_events[rows_gcs]
    charts_gcs = charts_gcs.pivot_table(index=['SUBJECT_ID', 'ICUSTAY_ID', 'CHARTTIME'], columns='CE_TYPE', values='VALUENUM')
    charts_gcs = charts_gcs.rename_axis(None, axis=1).reset_index()
    null_gcs_total = charts_gcs['GCS_TOTAL'].isnull()
    charts_gcs.loc[null_gcs_total, 'GCS_TOTAL'] = charts_gcs[null_gcs_total].GCS_EYE_OPENING + charts_gcs[null_gcs_total].GCS_VERBAL_RESPONSE + charts_gcs[null_gcs_total].GCS_MOTOR_RESPONSE
    charts_gcs =  charts_gcs[~charts_gcs['GCS_TOTAL'].isnull()]
    charts_gcs = charts_gcs.rename(columns={'GCS_TOTAL': 'VALUENUM'})
    charts_gcs['CE_TYPE'] = 'GCS_TOTAL'
    charts_gcs = charts_gcs.drop(columns=['GCS_EYE_OPENING', 'GCS_VERBAL_RESPONSE', 'GCS_MOTOR_RESPONSE'])

    # merge back with rest of the table
    rows_others = ~rows_bmi & ~rows_gcs
    charts = pd.concat([charts_bmi, charts_gcs, chart_events[rows_others]], ignore_index=True, sort=False)
    charts = charts.drop(columns=['SUBJECT_ID'])
    charts = charts.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False])
    return charts

def load_output_events(output_dir: str):
    print('loading output events')
    output_events = pd.read_pickle(os.path.join(output_dir, 'output_events_reduced.pkl'))
    print('loaded output events')
    return output_events

def create_categorical_variables(charts: pd.DataFrame, outputs: pd.DataFrame):
    df = pd.concat([charts, outputs], ignore_index=True, sort=False)
    df = df.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False])

    # bin according to OASIS severity score
    heart_rate_bins = np.array([-1, 32.99, 88.5, 106.5, 125.5, np.Inf])
    respiratory_rate_bins = np.array([-1, 5.99, 12.5, 22.5, 30.5, 44.5, np.Inf])
    body_temperature_bins = np.array([-1, 33.21, 35.93, 36.39, 36.88, 39.88, np.Inf])
    mean_bp_bins = np.array([-1, 20.64, 50.99, 61.32, 143.44, np.Inf])
    fraction_inspired_oxygen_bins = np.array([-1, np.Inf])
    gcs_total_bins = np.array([-1, 7, 13, 14, 15])
    bmi_bins = np.array([-1, 15, 16, 18.5, 25, 30, 35, 40, 45, 50, 60, np.Inf])
    urine_output_bins = np.array([-1, 670.99, 1426.99, 2543.99, 6896, np.Inf])
    bins = [heart_rate_bins, respiratory_rate_bins, body_temperature_bins, mean_bp_bins, fraction_inspired_oxygen_bins, gcs_total_bins, urine_output_bins]

    # labels
    heart_rate_labels = ['CHART_HR_m1', 'CHART_HR_n', 'CHART_HR_p1', 'CHART_HR_p2', 'CHART_HR_p3']
    respiratory_rate_labels = ['CHART_RR_m2', 'CHART_RR_m1', 'CHART_RR_n', 'CHART_RR_p1', 'CHART_RR_p2', 'CHART_RR_p3']
    body_temperature_labels = ['CHART_BT_m3', 'CHART_BT_m2', 'CHART_BT_m1', 'CHART_BT_n', 'CHART_BT_p1', 'CHART_BT_p2']
    mean_bp_labels = ['CHART_BP_m3', 'CHART_BP_m2', 'CHART_BP_m1', 'CHART_BP_n', 'CHART_BP_p1']
    fraction_inspired_oxygen_labels = ['CHART_VENT']
    gcs_total_labels = ['CHART_GC_m3', 'CHART_GC_m2', 'CHART_GC_m1', 'CHART_GC_n']
    bmi_labels = ['CHART_BM_m3', 'CHART_BM_m2', 'CHART_BM_m1', 'CHART_BM_n', 'CHART_BM_p1', 'CHART_BM_p2', 'CHART_BM_p3', 'CHART_BM_p4', 'CHART_BM_p5', 'CHART_BM_p6', 'CHART_BM_p7']
    urine_output_labels = ['CHART_UO_m3', 'CHART_UO_m2', 'CHART_UO_m1', 'CHART_UO_n', 'CHART_UO_p1']
    labels = [heart_rate_labels, respiratory_rate_labels, body_temperature_labels, mean_bp_labels, fraction_inspired_oxygen_labels, gcs_total_labels, urine_output_labels]
    # Chart event types
    ce_types = ['HEART_RATE', 'RESPIRATORY_RATE', 'BODY_TEMPERATURE', 'MEAN_BP', 'FRACTION_INSPIRED_OXYGEN', 'GCS_TOTAL', 'URINE_OUTPUT']

    return df, ce_types, labels, bins

def process_merge_chart_output(data_dir: str, output_dir: str):
    chart_events = load_chart_events_table(output_dir)
    chart_events = compute_bmi_gcs(chart_events)
    output_events = load_output_events(output_dir)
    df, ce_types, labels, bins = create_categorical_variables(chart_events, output_events)

    df_list = []
    df_list_last_only = []

    for ce_type, label, bin in zip(ce_types, labels, bins):
        # get chart events of a specific ce_type
        tmp = df[df['CE_TYPE'] == ce_type]
        # bin them and sort
        tmp['VALUECAT'] = pd.cut(tmp['VALUENUM'], bins=bin, labels=label)
        tmp = tmp.drop(columns=['CE_TYPE', 'VALUENUM'])
        tmp = tmp.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False])
        # remove consecutive duplicates
        tmp = tmp[(tmp[['ICUSTAY_ID', 'VALUECAT']] != tmp[['ICUSTAY_ID', 'VALUECAT']].shift()).any(axis=1)]
        df_list.append(tmp)
        # for logistic regression, keep only the last measurement
        tmp = tmp.drop_duplicates(subset='ICUSTAY_ID')
        df_list_last_only.append(tmp)

    df = pd.concat(df_list, ignore_index=True, sort=False)
    df.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False])
    df.drop_duplicates()

    print('Saving Reduced Charts + Outputs table')
    df.to_pickle(os.path.join(output_dir, 'charts_outputs_reduced.pkl'))
    df.to_csv(os.path.join(output_dir, 'charts_outputs_reduced.csv'), index=False)

    # for logistic regression
    df_last_only = pd.concat(df_list_last_only, ignore_index=True, sort=False)
    df_last_only = df_last_only.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'], ascending=[True, False])
    df_last_only.to_pickle(os.path.join(output_dir, 'charts_outputs_last_only.pkl'))
    df_last_only.to_csv(os.path.join(output_dir, 'charts_outputs_last_only.csv'), index=False)
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

    process_merge_chart_output(data_dir, output_dir)