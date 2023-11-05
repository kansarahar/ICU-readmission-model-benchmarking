import os
import argparse

from process_patient_admissions import process_patient_admissions
from process_diagnoses_procedures import process_diagnoses_procedures
from process_reduce_output_events import process_reduce_output_events
from process_reduce_chart_events import process_reduce_chart_events
from process_merge_chart_output import process_merge_chart_output
from process_charts_prescriptions import process_charts_prescriptions
from process_create_arrays import process_create_arrays

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

    print('------ PROCESSING PATIENT ADMISSIONS ------')
    process_patient_admissions(data_dir, output_dir)
    print('------ PROCESSED PATIENT ADMISSIONS ------')
    
    print('------ PROCESSING DIAGNOSES AND PROCEDURES ------')
    process_diagnoses_procedures(data_dir, output_dir)
    print('------ PROCESSED DIAGNOSES AND PROCEDURES ------')
    
    print('------ PROCESSING OUTPUT EVENTS ------')
    process_reduce_output_events(data_dir, output_dir)
    print('------ PROCESSED OUTPUT EVENTS ------')
    
    print('------ PROCESSING CHART EVENTS ------')
    process_reduce_chart_events(data_dir, output_dir)
    print('------ PROCESSED CHART EVENTS ------')
    
    print('------ MERGING OUTPUT AND CHART EVENTS ------')
    process_merge_chart_output(data_dir, output_dir)
    print('------ MERGED OUTPUT AND CHART EVENTS ------')
    
    print('------ PROCESSING CHART PRESCRIPTIONS ------')
    process_charts_prescriptions(data_dir, output_dir)
    print('------ PROCESSED CHART PRESCRIPTIONS ------')
    
    print('------ GENERATING TRAINING/TESTING/VALIDATION DATA ------')
    process_create_arrays(data_dir, output_dir)
    print('------ GENERATED TRAINING/TESTING/VALIDATION DATA ------')
    
