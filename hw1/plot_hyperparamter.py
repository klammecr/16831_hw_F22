from cProfile import label
from matplotlib import pyplot as plt

x = list(range(32, 256, 32))
y = [39.07623291015625, 302.5458068847656, 402.6897277832031, 150.611328125, 1507.9114990234375, 576.480224609375, 751.6830444335938]
y_std = [89.3756332397461, 302.98382568359375, 244.56858825683594, 330.3768005371094, 861.7947387695312, 204.622314453125, 414.1293029785156]
plt.plot(x, y, "rx-", label= "Average Return")
plt.plot(x, y_std, "kx--", label="Std Return")
plt.legend(loc="upper left")
plt.xlabel("Size of network per layer (nodes)")
plt.title("Effect of Network Size on Return")

plt.show()
pass