Meteorological data from radar and satellite sources often contain noise due to air turbulence and device manipulation, leading to inaccuracies in predictions. This work utilizes the netCDF4 library to import and manipulate meteorological data from NASA's TRMM_3B42_Daily_7 dataset for precipitation prediction. 

#use python=3.10
conda create -n rain_pred python=3.10
conda activate rain_pred
pip install -r requirments.txt
python main.py
python ANN_RE.py
