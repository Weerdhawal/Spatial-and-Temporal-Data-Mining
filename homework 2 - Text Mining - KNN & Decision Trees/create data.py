import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("final.csv", encoding = "ISO-8859-1")
df.rename(columns=df.iloc[0]).drop(df.index[0])
df['mainCATEGORIES']=df['mainCATEGORIES'].astype(str)
print(df.dtypes.eq(object))
c = df.columns[df.dtypes.eq(object)]

#df[c] = df[c].apply(pd.to_numeric, errors='coerce', axis=0)
#pd.to_numeric(df['mainCATEGORIES'], errors='coerce').fillna(0)
#df.fillna(0)
scaler = StandardScaler()
print(df.dtypes.eq(object))
scaler.fit(df.drop('mainCATEGORIES', axis = 1))
#print(df)