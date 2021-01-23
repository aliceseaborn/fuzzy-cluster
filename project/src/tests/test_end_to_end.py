# -------------------- CONFIGURE ENVIRONMENT -------------------- #

# Standard math libraries
import pandas as pd

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Configure paths
from pathlib import Path
data_path = Path('Data')

# Custom library
import FuzzyCluster as FC

# Import from CSV
Data = pd.read_csv(data_path / 'SyntheticData.csv')

# -------------------- SETUP PREFERENCES -------------------- #

# Omega bias preferences
Omega = [ 0.30, 0.30, 0.40 ]

# Set big S
S = 75

# Verbosity
verb = False

# Copies of the dataset
Master = Data.copy()
Deck   = Data.copy()

# ------------------------- GLOBAL GROUPS ------------------------- #

# Template columns
grp_col = [ 'Group Number', 'Record Number', 'Centroid Flag' ]

# Form data for simulation
Groups = pd.DataFrame( columns = grp_col )

# -------------------- GROUPS DATAFRAME -------------------- #

# Pop from deck
card = FC.iterator( Deck, True, False )

# Append card to Groups
Groups.loc[0] = [ 0, card.name, 1 ]

# -------------------- EXECUTE ALGORITHM -------------------- #

Results = FC.fuzzy_cluster( Master, Deck, Groups, Omega, S, verb )

# -------------------- EXPORT RESULT SET -------------------- #

Results.to_csv("ResultSet.csv", sep=',')




