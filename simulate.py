
import matplotlib.pyplot as plt

import simulations
import simulations.social

#simulations.simulate_with_distribution("Exponential")
#simulations.simulate_with_distribution("Power_Law")

#simulations.check_mixed_exps()

fig, ax = plt.subplots()
for i in range(10):
    print(f"Social simulation, iteration {i}")
    simulations.social.social_sync_simulation(fig, ax)

plt.show()
