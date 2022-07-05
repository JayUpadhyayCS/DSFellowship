import pandas as pd

data_xls = pd.read_excel('InSAR_data_south/20190122.xlsx', 'Sheet2', index_col=None)
data_xls.to_csv('csvfile.csv', encoding='utf-8', index=False)