'''
In this file:

'''
import sys
from pathlib import Path
import pandas as pd
import numpy as np
np.random.seed(42)

# PATHS - using relative paths from project root
PERSONAL = Path("/Users/cestrarz./PythonLearning")
CODE_DIR = PERSONAL / "code"
DATA = PERSONAL / "data"
OUTPUT = PERSONAL / "output"

# CREATE LOG
sys.path.append(str(PERSONAL / "python"))
from utils.logging_utils import start_logging
logger = start_logging(log_file="starting_template.log",
                       log_dir=PERSONAL/"output/logs")

########################################################
# MAIN SCRIPT 
########################################################
