from data_provider import build_train_data
import pandas as pd
import numpy as np

data = build_train_data("WIKI/AAPL")

x = data["X"]
# x['Y'] = data['Y']
from sklearn.decomposition import PCA

N_COMP = 30
pca = PCA(n_components=N_COMP)
pca.fit(x)

new = pd.DataFrame(pca.transform(x), columns=["PCA%i" % i for i in range(N_COMP)], index=x.index)
# print(new)
with pd.option_context("display.max_rows", None, "display.max_columns", 3):
    new["Y"] = data["Y"]
    print(x.apply(lambda y: y.corr(new["Y"])).sort_values())
    print(new.apply(lambda y: y.corr(new["Y"])).sort_values())

# print(x_and_labels.tail(50))
