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
sample_data = pd.read_csv(DATA/"sample_data.csv")
print(sample_data.head())

for i in range(5):
    if i % 2 == 0:
        print(f"{i} is EVEN")
    else:
        print(f"{i} is ODD")


def scream():
    print("AAAAH")


scream()


rands = [1, 23, 22]
rand0 = rands.append(3)
newarray = np.array(rands)
newarray += 1
print(newarray)
