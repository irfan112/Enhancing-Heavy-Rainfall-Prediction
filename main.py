import numpy as np
from netCDF4 import Dataset
import pandas as pd
from tqdm import tqdm
import warnings
import os
import json
from datetime import datetime, timedelta
# from multiprocessing import Pool
warnings.filterwarnings("ignore")
from ANN_RE import ann_re, ann_pre
import torch
from ANN_RE_model import ANN
from ANN_pre_model import ANN_2
from sklearn.preprocessing import MinMaxScaler


output_directory = 'csv_Conversion'
nc_files_directory = 'D:/CLoud_seeding/backupdata'

def parse_filename(filename):
    if '(' not in filename:
        parts = filename.split('.')
        year = int(parts[1][:4])
        month = int(parts[1][4:6])
        day = int(parts[1][6:8]) 
        time_suffix = int(parts[2])

        reference_datetime_str = str(year)+' '+str(month)+' '+str(day)+' '+'05:00:00'
        reference_datetime = datetime.strptime(reference_datetime_str, '%Y %m %d %H:%M:%S')
        current_datetime = reference_datetime + timedelta(hours=time_suffix)
        print("current_datetime: ", current_datetime)
    
    return current_datetime, time_suffix

def processing_nc_file(filename):
    print("Processing : ", filename)

    if filename.endswith(".nc4"):
        nc_file = os.path.join(nc_files_directory, filename)
        nc_basename = os.path.basename(nc_file [:-4])
        if nc_basename.startswith('3B42_Daily'):
            current_datetime, time_suffix = parse_filename(nc_basename)
        nc = Dataset(nc_file, 'r')
                
        precipitation_values = nc.variables['precipitation'][:]
        HQprecipitation_values = nc.variables['HQprecipitation'][:]
        IRprecipitation_values = nc.variables['IRprecipitation'][:]
        randomError_values = nc.variables['randomError'][:]

        nlat_values = nc.variables['lon'][:]
        elon_values = nc.variables['lat'][:]
  
        y_dim = nlat_values.size
        x_dim = elon_values.size
        
        print("y_dim: ", y_dim)
        print("x_dim: ", x_dim)

        x_list = []
        y_list = []
        precipitation_list = []
        HQprecipitation_list = []
        IRprecipitation_list = []
        randomError_list = []

        for y_index in tqdm(range(y_dim)):
            for x_index in range(x_dim):
                nlat = nlat_values[y_index]
                elon = elon_values[x_index]
                
                precipitation = precipitation_values[y_index, x_index]
                if precipitation >= 2:
                    x_list.append(elon)
                    y_list.append(nlat)
                    HQprecipitation = HQprecipitation_values[y_index, x_index]
                    IRprecipitation = IRprecipitation_values[y_index, x_index]
                    randomError = randomError_values[y_index, x_index]
                    
                    precipitation_list.append(precipitation)
                    HQprecipitation_list.append(HQprecipitation)
                    IRprecipitation_list.append(IRprecipitation)
                    randomError_list.append(randomError)
                else:
                    continue
        
        data = {
                
                'Datatime':f'{current_datetime.year}_{current_datetime.month}_{current_datetime.day}_{current_datetime.hour}', \
                'nlat': y_list, 'elon': x_list, 'Precipitation': precipitation_list, \
                'HQprecipitation': HQprecipitation_list,  \
                'IRprecipitation' : IRprecipitation_list, \
                'randomError': randomError_list, 
                
                }
        formatted_datetime = current_datetime.strftime('%Y_%m_%d_%H')
        
        df = pd.DataFrame(data)
        filename = "precipitationDATA.csv"
        file_path = os.path.join(output_directory, filename)
        file_exists = os.path.exists(file_path)
        # print("The CSV File Exists: ", file_exists)
        if file_exists != True: 
            df.to_csv(os.path.join(output_directory , f'precipitationDATA.csv'), index=False)
            ann_re.training_ANN_RE(df)
            ann_pre.taining_ANN_Pre(df)
            
        else:
            model = ANN()
            # print("ANN For Error Matigation Error! ", model)
            model.load_state_dict(torch.load("ANN_RE_model.pt"))
            model.eval()
            
            data_training = df.iloc[:,3:].copy()
            data_training_ = data_training.replace('--', 0)
            data_training__ = data_training_.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
            data_training__ = data_training__.dropna()
        
            data_training_cleaned = data_training__.copy()
            
            X = data_training_cleaned[['Precipitation', 'HQprecipitation', 'IRprecipitation', 'randomError']]            
            X_np = X.to_numpy()
            
            scaler = MinMaxScaler()
            X_np_normalized = scaler.fit_transform(X_np)

            input_tensor = torch.tensor(X_np_normalized, dtype=torch.float32) 
            
            with torch.no_grad():
                predictions = model(input_tensor)
                # print("predictions: ", predictions)
                
            precipatation_ANN_model1 = predictions
             
            model2 = ANN_2()   
            model2.load_state_dict(torch.load("ANN_pre_model.pt"))
            model2.eval()

            
            Xnp1 = precipatation_ANN_model1

            scaler = MinMaxScaler()
            Xnp1_normalized = scaler.fit_transform(Xnp1)

            input_tensor1 = torch.tensor(Xnp1_normalized, dtype=torch.float32) 
            with torch.no_grad():
                predictions1 = model2(input_tensor1)
                
            
            predictions1_df = pd.DataFrame(predictions1.numpy())
            predictions1_df.columns = [formatted_datetime]

            output_csv_path = os.path.join(output_directory, 'lstm_test.csv')
            if os.path.exists(output_csv_path):
                existing_df = pd.read_csv(output_csv_path)
                combined_df = pd.concat([existing_df, predictions1_df], axis=1)
                combined_df = combined_df.iloc[:697, :]
                combined_df.to_csv(output_csv_path, index=False)
                # combined_df = combined_df.iloc[:5, :]
            else:
                predictions1_df = predictions1_df.iloc[:697, :]
                predictions1_df.to_csv(output_csv_path, index=False)
            
            nc.close()

def main():
    file_list = sorted(os.listdir(nc_files_directory))
    for i in file_list:
        processing_nc_file(i)

if __name__ == '__main__':
    main()