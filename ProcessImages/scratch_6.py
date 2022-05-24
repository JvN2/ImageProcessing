import numpy as np
import pandas as pd


# arr = np.linspace(0, 10, 5)
# df = pd.DataFrame(arr.tolist(), columns=['test'])
# df = pd.DataFrame()
# # df = df.merge(df, pd.DataFrame(arr))
# # df['test_i2'] = np.random.random_integers(0,10, 10)
# df.at[:,'test4'] =arr
# print(df)
column_names = ['test', 'ttry']
column_names += [f'{i:.0f}: test' for i in np.linspace(0, 300, 50)]
print(column_names)
# print(sorted(column_names, key=lambda x: int(x[:x.index(':')])))