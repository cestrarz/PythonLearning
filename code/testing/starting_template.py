import sys
from pathlib import Path

# PATHS
PERSONAL = Path("/Users/cestrarz./PythonLearning")
CODE_DIR = PERSONAL/"code"
DATA = PERSONAL/"data"
OUTPUT = PERSONAL/"output"

sys.path.append(str(PERSONAL/"python")) # to search for logging_utils in PERSONAL/utils

import numpy as np
import pandas as pd
from utils.logging_utils import setup_logging

# CREATE LOGGER
logger = setup_logging(log_file="starting_template.log")
logger.info("Starting script")
np.random.seed(42)
########################################################

sample_data = pd.read_csv(DATA/"sample_data.csv")
print(sample_data.head())



########################################################
# Close logging
logger.info("Script completed")
