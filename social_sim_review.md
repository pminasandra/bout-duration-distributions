## Scope

I've reviewed all and tested most of the simulation related code.

I haven't reviewed or tested the plotting related code because if my design commments are incorperated, that will make those sections much easier to read and test.

## Comments

### Small changes to text
* Appendix doesn't specify number of simulations or number of bouts per simulation that I can see. Would be helpful to include that as well as the equation for the relationship between proportion in the state and probability of state change.
* You can increase the generality of your discussion without chaning your simulation by consider "state A" vs "all other states" instead of all other states.
* Per our discussion, you can mention that although theoretically with a large population size, the a population subject to this model would land in one of the steady state, in practice with 10 individuals, no steady state a single proportion is reached.

### Code design
Everything stated in the "optimizing for readability" section applies here as well. In addition, it would be nice to save the output data of the simulation, not just the plots so that you have the option to re-analyze and plot it multiple ways without rerunning the simulation. You do that in the error rate simulation and it would be good to continue that practice here.