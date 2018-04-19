import pandas as pd
import numpy
from sklearn.preprocessing import MinMaxScaler



df = pd.read_csv("checkfinal_nc.csv",index_col=False)
#scalar = MinMaxScaler()
#print(scalar.fit(df))
#df=scalar.transform(df)
print(df)

#numpy.savetxt('normed.csv',df,delimiter=',')