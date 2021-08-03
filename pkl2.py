import pickle
import pandas as pd
f = open('result.pkl','rb')
data = pickle.load(f)
pd.set_option('display.width',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_colwidth',None)
print(data)
inf=str(data)
ft = open('test1.csv', 'w')
ft.write(inf)