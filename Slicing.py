import numpy as np
import os
#Since the tensor is of shape 97 by 33 and we want a 96 by 32 the most inner part of the matrix is sliced 
#Now they are perfect for Deep Learning, because of the powers of two

# Define the directory where the data is located
data_dir = 'traindatastress'

# Loop over all 60 samples
for i in range(1, 61):
    # Load the .npy file from the folder
    file_name = os.path.join(data_dir, f'train_data_{i}.npy')
    data = np.load(file_name)
    
    # Slice the data to remove the first row and first column
    datasliced = data[1:, 1:]

    # Save the sliced data to a new .npy file in the same folder
    sliced_file_name = os.path.join(data_dir, f'train_data_{i}_sliced.npy')
    np.save(sliced_file_name, datasliced)

    print(f"Saved {sliced_file_name}")
