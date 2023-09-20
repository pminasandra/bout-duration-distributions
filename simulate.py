
import matplotlib.pyplot as plt

import config
import utilities

import simulations
import simulations.social

if not config.SUPPRESS_INFORMATIVE_PRINT:
    print = utilities.sprint

#simulations.simulate_with_distribution("Exponential")
#simulations.simulate_with_distribution("Power_Law")

#simulations.check_mixed_exps()

fig, ax = plt.subplots()
for i in range(10):
    print(f"Social simulation, iteration {i}")
    simulations.social.social_sync_simulation(fig, ax)

ax.set_xlabel("Time since start of bout")
ax.set_ylabel("Hazard rate")
utilities.saveimg(fig, "social_reinforcement_simulation")
