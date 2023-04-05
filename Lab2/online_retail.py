import pandas as pd
from matplotlib import pyplot as plt
from apyori import apriori

df = pd.read_csv("OnlineRetail.csv", encoding="unicode_escape")

true_cols = df.columns.tolist()

cols = true_cols[2:4]
cols.append(true_cols[7])
df = df[cols]
df_values = df.values[:, :]

items = []
for i in range(0, 541909):
    items.append([str(df_values[i, j]) for j in range(0, 3)])
final_rule = apriori(items, min_support=0.001, min_confidence=0.80)
final_result = list(final_rule)

# print(final_result)

for x in final_result:
    if ((str(x.ordered_statistics[0].items_base)[9:]) != "()" and (str(x.ordered_statistics[0].items_base)[9:]) != "({'nan'})") :
        confidence = str(x.ordered_statistics[0].confidence)
        string = str(x.ordered_statistics[0].items_base)[9:]
        string2 = str(x.ordered_statistics[0].items_add)[9:]
        print(string, "-->", string2, "conf:", confidence)
