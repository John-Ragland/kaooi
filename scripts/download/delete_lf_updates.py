'''
this script deletes lf files up to certain date
'''

import os
import glob
from datetime import datetime

# Specify the path pattern for the files to be checked
path_pattern = '/gscratch/coenv/jhrag/data/kauai/transmissions/ooi_lf/*.nc'

# Specify the valid date till when files should be kept (inclusive)
# Format: 'YYYYMMDD'
valid_till = '20240326'

# Convert valid_till to a datetime object for comparison
valid_till_date = datetime.strptime(valid_till, '%Y%m%d')

# Loop through all files matching the path pattern
for file_path in glob.glob(path_pattern):
    # Extract the date part from the file name
    # Assuming file name format is 'yyyymmddThhmmss.nc'
    file_date_str = os.path.basename(file_path).split('T')[0]
    
    # Convert the file date to a datetime object
    file_date = datetime.strptime(file_date_str, '%Y%m%d')
    
    # Check if the file date is after the valid till date
    if file_date > valid_till_date:
        # Delete the file
        os.remove(file_path)
        print(f"Deleted: {file_path}")