import  pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

data = {"MPV-MCTS": [0.981472, 0.980378, 0.983331, 0.984259, 0.984521],
        "Meta MPV-MCTS": [0.979198, 0.981437, 0.983818, 0.984359, 0.984617],
        "Serial": [1,2,3,4,5]}

df = pd.DataFrame(data, columns= ["MPV-MCTS", "Meta MPV-MCTS", "Serial"])

# df.plot.line(x='Serial', y=["MPV-MCTS", "Meta MPV-MCTS"], color={"MPV-MCTS":"red", "Meta MPV-MCTS":"blue"})

fig = plt.figure()
ax = plt.axes()

plt.plot(df["Serial"], df["MPV-MCTS"], label="MPV-MCTS", marker='o')
plt.plot(df["Serial"], df["Meta MPV-MCTS"], label="Meta MPV-MCTS", marker='o')
plt.legend()
plt.show()