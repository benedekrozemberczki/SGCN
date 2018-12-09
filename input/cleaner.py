import pandas as pd

tools = pd.read_csv("clean_bitcoin_otc.csv")
tools = tools.values.tolist()
curr = set()
for t in tools:
    curr = curr.union([t[0],t[1]])

curr = list(curr)
mapper= {c:ind for ind, c in enumerate(curr)}
tools = [[mapper[t[0]],mapper[t[1]],t[2]] for t in tools]
tools = pd.DataFrame(tools, columns = ["id1","id2","sign"])
tools.to_csv("bitcoin_otc.csv", index = None)
