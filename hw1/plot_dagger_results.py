from matplotlib import pyplot as plt

# Ant
iteration = list(range(10))
# returns     = [4684.7060546875, 4586.939453125, 4700.583984375, 4722.8984375, 4763.44921875, 4798.826171875, 4691.6171875, 4745.880859375, 4227.6650390625, 4724.25732421875]
# returns_std = [143.7537078857422, 674.8822021484375, 92.76825714111328, 100.55745697021484, 112.72573852539062, 78.77670288085938, 70.92704010009766, 201.52789306640625, 1161.0738525390625, 97.85667419433594]
# bc_returns  = [4684.7061] * 10
# expert_return = [4739.10] * 10

# # Ant plot
# #plt.plot(iteration, returns, "kx-", label= "Average Return")
# plt.xlabel("Iteration")
# plt.ylabel("Return from Policy")
# plt.errorbar(iteration, returns, returns_std, label = "DAgger")
# plt.plot(iteration, expert_return, label = "Expert")
# plt.plot(iteration, bc_returns, label="Behavior Cloning")
# plt.title("Performance of Learning Methods on OpenAI Gym Ant Environment")
# plt.ylim(2500, 7000)
# plt.legend(loc="lower right")
# plt.show()


# Walker 2D
returns     = [302.5458068847656, 5246.00390625, 4443.453125, 4537.498046875, 5336.828125, 5373.2978515625, 5335.40966796875, 5304.00244140625, 5392.7900390625, 5324.2216796875]
returns_std = [302.98382568359375, 36.138431549072266, 935.8585815429688, 1048.186279296875, 23.544864654541016, 30.591964721679688, 43.868404388427734, 78.97154998779297, 32.891056060791016, 35.62556076049805]
bc_returns  = [302.5458] * 10
expert_return = [5347.19] * 10

plt.xlabel("Iteration")
plt.ylabel("Return from Policy")
plt.errorbar(iteration, returns, returns_std, label = "DAgger")
plt.plot(iteration, expert_return, label = "Expert")
plt.plot(iteration, bc_returns, label="Behavior Cloning")
plt.title("Performance of Learning Methods on OpenAI Gym Walker 2D Environment")
plt.ylim(0, 7000)
plt.legend(loc="upper right")
plt.show()
pass