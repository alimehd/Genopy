import pandas as pd
df = pd.read_csv('/Users/alimehdi/Desktop/COVID-19 Dataset/samples_matrix.tsv', sep='\t') # loading the file as a dataframe
df = df.loc[:, (df != 0).any(axis=0)] # removing all zero columns
print("Shape of the matrix", df.shape)
df.head()
