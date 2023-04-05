import pandas as pd
from matplotlib import pyplot as plt
from apyori import apriori

df = pd.read_csv("titanic.csv")
df_values = df.values[:, 1:]

items = []
for i in range (0, 2200):
    items.append([str(df_values[i,j]) for j in range (0,4)])
final_rule = apriori(items, min_support=0.005, min_confidence=0.8, min_length=2, min_lift = 1.2)
final_result = list(final_rule)

for x in final_result:
    if x.ordered_statistics[0].items_add.issuperset(["Yes"]):
        confidence = str(x.ordered_statistics[0].confidence)
        string = str(x.ordered_statistics[0].items_base)[9:]
        string2 = str(x.ordered_statistics[0].items_add)[9:]
        print(string, "-->", string2, "conf:", confidence)

        plt.scatter(x.support, x.ordered_statistics[0].confidence, alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()
