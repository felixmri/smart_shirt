import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


s1_1 = pd.read_csv('data2/caugh.csv', header=None)  # first subject, sensor 1
ch1 = s1_1.iloc[:, 1]
ch2 = s1_1.iloc[:, 2]
r1_1_ch1 = (-ch1 * 100) / (ch1 - 1)  # channel one (normal breathing)
r1_1_ch2 = (-ch2 * 100) / (ch2 - 1)  # channel two (normal breathing)

time = np.arange(0, .015 * len(r1_1_ch1), .015)  # 15 ms per time-point

s1_2 = pd.read_csv('data2/no_cough.csv', header=None)  # first subject, second trial
ch1 = s1_2.iloc[:, 1]
ch2 = s1_2.iloc[:, 2]
r1_2_ch1 = (-ch1 * 100) / (ch1 - 1)  # channel one (coughing)
r1_2_ch2 = (-ch2 * 100) / (ch2 - 1)  # channel two (coughing)



fig, axs = plt.subplots(4, 1)
fig.suptitle('Subject 1')
axs[0].plot(time, r1_1_ch1)
axs[0].set_ylabel("Channel 1")
axs[1].plot(time, r1_1_ch2)
axs[1].set_ylabel("Channel 2")
axs[2].plot(time, r1_2_ch1)
axs[2].set_ylabel("Channel 1")
axs[3].plot(time, r1_2_ch2)
axs[3].set_ylabel("Channel 2")
plt.show()