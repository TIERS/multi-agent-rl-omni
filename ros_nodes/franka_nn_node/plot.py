import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# baseline
df = pd.read_csv("e3_p1_m2.csv")
distances = df["distance_from_target"].to_numpy()
elapsed_secs = df["elapsed_time"].to_numpy()

# plot the distance from target over time with sns
sns.set_theme(style="darkgrid")
sns.lineplot(x=elapsed_secs, y=distances)
# add labels
plt.xlabel("Time (s)")
plt.ylabel("Distance from target (m)")
# add title
plt.title("Distance from target over time")
plt.show()