import h5py
import numpy as np
import re

src = "NiO_GW.h5"

with h5py.File(src, "r") as source_file:
    # 1. Find all groups that follow the 'iter{number}' pattern
    # We use a list comprehension to extract the integers from the keys
    iterations = []
    for key in source_file.keys():
        if key.startswith("iter"):
            try:
                # Extract the numeric part (e.g., 'iter14' -> 14)
                it_num = int(re.search(r'\d+', key).group())
                iterations.append(it_num)
            except (AttributeError, ValueError):
                continue

    if not iterations:
        raise ValueError(f"No 'iter' groups found in {src}")

    # 2. Identify the last iteration
    last_it = max(iterations)
    dst = f"NiO_GW_iter{last_it}.h5"
    
    print(f"Detected last iteration: {last_it}. Creating {dst}...")

    # 3. Perform the copy
    with h5py.File(dst, "w") as dest_file:
        # Copy the group 'iterN' to the new file
        source_file.copy(f"iter{last_it}", dest_file)

        # Create the scalar dataset 'iter' as required by your reader
        dest_file.create_dataset("iter", data=np.int64(last_it))