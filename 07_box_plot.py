#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs')
import utilities as U
#-----------------------------------------------------------------------------------------#
import pandas as pd
#-----------------------------------------------------------------------------------------#

tabnet_df = pd.read_csv('evaluation/evaluation_metrics.csv')
U.plot_box(tabnet_df)

#-----------------------------------------------------------------------------------------#