import pandas as pd
import fcntl
import os
import jax


csv_file = "test.csv"





with open(csv_file + '.lock', 'w') as lock_file:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

    # Create new row with current results
    new_row = {
        'test': 1,
        'test2': 2,
    }
    
    # Try to load existing results dataframe if it exists
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=list(new_row.keys()))

    # Append new results
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save updated dataframe
    df.to_csv(csv_file, index=False)