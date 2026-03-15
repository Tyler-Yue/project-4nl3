import json
import os
import sys
import time

import numpy as np

input_dir = '/app/input_data/'
output_dir = '/app/output/'
program_dir = '/app/program'
submission_dir = '/app/ingested_program'

sys.path.append(program_dir)
sys.path.append(submission_dir)


def main():
    prediction = os.path.join(input_dir, 'prediction')
    np.savetxt(os.path.join(output_dir, 'prediction'), prediction)
    
    print(f'Ingestion Program finished.')


if __name__ == '__main__':
    main()
