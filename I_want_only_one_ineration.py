import h5py

# Open the original file in read mode
with h5py.File('NiO_GW.h5', 'r') as source_file:
    # Open the new file in write mode
    with h5py.File('NiO_GE_iter14.h5', 'w') as dest_file:
        # Copy the group 'group_to_extract' to the root of the new file
        source_file.copy('iter14', dest_file)