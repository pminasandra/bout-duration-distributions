---
title: Behavioural Inertia and its quantification
author: Pranav Minasandra
---

# What is behavioural inertia?

Behavioural inertia is here defined as the tendency of an animal to remain in its current behavioural state.
More rigorously, it is the instantaneous probability of an animal to not leave its current behavioural state.
If the animal has been in its current behaviour for time $t$, and we say that the current behavioural bout started
at 0 and ended at $T$, then behavioural inertia is

$$
BI(t) = \Pr(T > t + dt \mid T > t).
$$

This simplifies nicely to

$$
BI(t) = \frac{\Pr(t)dt}{\Pr(T>t)}
$$

Note that behavioural inertia is simply $1 - h(t)$, where $h(t)$ is the hazard function.


# Computing behavioural inertia from data

$BI(t)$ can directly be computed from the data using the following pseudocode:

```
[t1, t2, ..., tn] = all_bout_durations(data)
[T1, T2, ..., TN] = unique_bout_durations(data)

dt = epoch_length

for unique_bout_duration in [T1 ... TN]:
  ubq = unique_bout_duration
  BI(ubq) = count(t_i > ubq + dt) / count(t_i > ubq) for t_i in [t1 ... tn]
  store(BI(ubq))
```

