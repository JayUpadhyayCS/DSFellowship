import pandas as pd
excelPath='InSAR_data_south/20190122.xlsx'
csvPath='InSAR_data_south/CSV/'
data_xls = pd.read_excel(excelPath, 'Sheet2', index_col=None)
data_xls.to_csv(csvPath+'csvfile.csv', encoding='utf-8', index=False)
pd.