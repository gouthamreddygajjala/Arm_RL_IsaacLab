import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/darms/joint7_freeze_trace.csv")
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(df["step"], df["pos_rad"])
axes[0].set_ylabel("position (rad)")
axes[1].plot(df["step"], df["vel_rad_s"])
axes[1].set_ylabel("velocity (rad/s)")
axes[1].set_xlabel("step")
plt.show()          