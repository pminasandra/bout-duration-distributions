# Review Status

I have read and left comments on all the code in the simulations module that is used in the err simulation. All the comments below largely pertain to this simulation and may or may not apply to the others.

I have written and run tests on the Simulator object. I've started writing tests for the functions in the `simulations/__init__.py` file, but those are not yet thorougly tested.

Todo for me:

* Finish testing code in `simulations/__init__.py` module (assuming no major refactor is planned)
* Test code used in simulator.py itself
* Read and comment on the other two simulations
* Write tests for the code int he other two simulations

Out of scope/not planned:
* I don't plan to evaluate the code that zooms in on a particular part of the parameter space any more than I have already.
* I am considering out of scope evaluating code outside of the simulations module and simulate.py even if that code is called by the simulations module. I will assume such code is working for the purposes of my tests.


# Big Picture Topics

## Code Quality

Here, I don't expect you to make any changes to the code (unless you are doing a huge refactor for other reason), this feedback is more to keep in mind the next time you embark on a similar project.

### Optimize for Readability


In my mind, optimizing for readability would mean having the processes in the code mirror as closely as possible how researchers conceptualize the real proccess both in name and structure. It seems like there are a few processes you're simulating here:
1) generating the ground truth behavioral sequences from some process/distribution,
2) the conversion between the truth and some measurable values (you call them "features" here),
3) inferring observed behavioral sequence from measurements
4) inferring the generating process from inferred behavioral states. Within process 2 and 4 there are sub-processes where you convert back and forth between discrete and continuous sequences. I'd also add a step 5 of plotting the simlation results.

Your current code is something like this:
```
/simulate.py
for param_set in step_1_param_space; (params used in step 1 only)
    /simulations.simulate_with_distribution
    set error rate params (used in step 2)
    add some details to ground truth distribution
    for i in 1:nsimlations:
        //simlations._simulate_and_get_results
        set num bouts (param related to step 2 because it's the amount that we observe)
        simulations.simulateor.Simulator
            do step 1
            for param_set in step_2_param_space:
                do step 2 (Deleniation between 1 and 2 not very clean)
        do step 3 (no for loop here because always done the same way)
        do step 4 (no parametrized loop here because always done the same way)
    save summary to disk
step 5 (plotting)
```
For the for loops listed above, the iterations are independent so in theory they can be paralellized and one of them is.
A more readable structure would be something like this
```
universe_of_ground_dists_params = [....] # each item here is an object that fully specifies the variables needed to generate a ground truth sequence, these shouldn't need modification later to be used
universe_of_measurement_params = [...] # should include not just the err params, but also nbouts, epoch, etc other things related to the translation from ground truth to measurement
(opt) state_classification_params = [...] # this is if you want to change fitting.classifier
(opt) dist_inference_params = [...] # e.g if you want to test multiple models 
```

Then there a couple ways that I can think of that you can structure the code itself in readable way:

```
def run_study(a_set_of_ground_dist_params, a_set_of_measurement_params, ...):
    ground_truth_bout_lengths = generate_ground_truth(a_set_of_ground_truth_dist_params)
    measurements (or features) = observe(ground_truth_bout_lengths, measurement_params)
    inferred_states = infer_states(features)
    inferred_distributions = infer_distributions(inferred_states)
    return (infered_distributions)

results = {}
for gtp in universe_of_ground_truth_params:
    for mesp in universe_of_measurment_params:
        for i in range(nsims_per_param_set):
            results[(gtp, mesp, i)] = run_study(gtp, mesp)
plot_results(results)
```

or alternatively

```
all_ground_truth_data = generate_ground_truth(universe_of_ground_dist_params, nsim_per_param_set)
all_measurements = generate_measurements(universe_of_measurement_params, all_gorund_truth_data)
inferred_states = infer_states(all_measurments)
inferred_distributions = infer_distributions(inferred_states)
```

Of course, this isn't to stop you from paralellizing,
you can farm out any of the written (or implied inside a function) for loops into mp.
The point is to make functions/components named based on the real processes the represent.
Break implementation complexity into those components, then push it inside these readably named functions and objects. 
This will let you more easily iterate on each piece independently and also make the flow of data and execution flow more clear to the reader.
For example, when you're tyring to illustrate your specific case, the main thing you're trying to do explore a smaller parameter spece with more granularity in step 2 for one value of step 1, but because the logic is all mixed together between generating the parameter space, and executing the different steps, you had to copy a lot of the code and tweak it here and there instead of changing the in puts and reusing code. (You were able to reuse the some code where parts your were not changing were encapusated.)

### Optimizing for Performance

You would have to do some actual profiling to validate this,
but I think you make this much more performant by leveraging np/pandas operations more so that your loops get moved from python to c. 
The only part that I think cannot really be done with math-like numpy/pandas is step 4. Steps 1-3 I suspect you can do all with clever dataframe operations.
Especially converting back and forth between an array of bout lengths and an array of discrete states. You shouldn't need a while loop to do that.
It should be possible to create a couple of performant lower-level functions that take care for those operations with minimal python looping.

Also, as previously mentioned, you are doing way more random number generation than required and random number generation tends to be expensive. Even if you're not doing a fully analytical solution, think about where you *actually* need randomness/simulation vs where you just run an operation on a discretized PDF or CDF.

### Readability/Performance Tradeoff

A script of pandas/numpy operations is not readable. Doing a for loop or list comprehension (even essentially for loop farmed out to mp) is generally a bit more readable that a matrix-operation-like statement so there is a bit of a tradeoff here. A practical approach is probably to focus on readability for the design and sequencing of high level functions and performance for lower level functions. (To do this, each function that you create needs to be in a specific place on this heirachy, not mixing high level and low level elements.) Somewhere in the middle, these values will bump up against each other.

### Keep Labels/Metadata with Data

There's several places where you pass around unlabeled lists and rely on the order to later apply a labeling. The most aggregious example of this is the `fit_results` and `fit_results_spec`. This is a list of lists. The ordering of the outer list doesn't matter, but the ordering of the inner list matters a *lot* as each element corresponds to as set of error params. It's very difficult to validate whether the ordering is properly preserved through each function and loop that generates these objects. Therefore, it's hard to develop confidence that how you've labeled the final graph actually corresponds to the parameters used to generate the data.
    
    
## Conceptual/Design Feedback

### Assumptions
Here are some assumptions that I think may impact the results. IMO it would be good to one of the following with each of them:
* actually if they matter
* explain in the text (at least in the appendix) that there's some logical reason that they wouldn't impact the results
* acknowledge in the text that this is an untested assumption

Here's my list:
* Only two states
* Distribution is the same for both states
* Error rate is the same for both states

### Does the fact that this is a sequence of multiple bouts even matter?

Part of what makes this simulation neccessary/hard to do with math is the interaction between errors at the boundary between bouts. e.g. if the last timepoint of an A bout is misclassified as a B timepoint it both makes the A bout appear shorter and the adjacent B bout appear longer. But I'm wondering...is this effect even big enough to be relevant? If the bouts are sufficiently long relative to length of a timepoint (epoch?) and error rate suffciently low, then this definitely have only a neglible impact. In that case, you wouldn't need to generate behavioral "sequences" at all. You would just need to generate distributions of bout lengths with and without error. Lining them up into a sequence with bouts of other states would be a unnecessary exercise. 

Of course, if the data is sufficiently high grain releative to the bout length or error rate sufficiently high, the impact of misclassication on the end of one bout "extending" the adjacent bout would have an impact.

Have you thought about this? Do you have some evidence that we are in the second case within the parameter space you're exploring?

Of course, lining the bouts up next to each other isn't going to hurt...it just makes the simulation unnecessarily complex. It if you save the records from your simulation, it should be possible to empircally evaluate which case we're in. Might be worth checking.

### Normal dist isn't doing anything
Conceptually, with your two state simulation if you isolate the process of going form a discretized sequence that represents true behaviors to a descritized sequence that represents observed/inferred behaviors (e.g. AAAAABBBBB -> AABAABABBB), the only relevant parameters are P(Bobs|Atrue) and P(Aobs|Btrue) which, in practice, you've set to be the same value. You have encoded these two paramters, and explained them in an unnecessarily complex way. What you describe in the paper and implement in the code is something like calulating muA of a normal distribution such that cnorm(0, muA, 1) = P(Bobs|A), then sample from this distribution, then collapse it back down by just checking whether the sample is < 0. This process could be simplified by just doing a weighted coinflip of P(Bobs|A) to convert between true and observed/inferred sequence without a mediating variable/feature. This would simplify both your code and your text. Because these two processes are equivalent, I recommend at least simplifying your text even if you don't want to take the time to revise and rerun your code.

### Analysis Suggestion - Empirical Comparison of Dist with and w/o Error

Before jumping to running a model to infer the distribution of bout lengths with error, it might be useful to directly compare your the (synthetic) true bout lenghths and your bout lengths with err. ei generate and empirical/discrete PDF (ie histogram) and CDF of each overlay them. You can see directly to what extent bout lengths are shortened.


## Clarifications

* How is the claim that error impacting inferred distribution implies super long bouts supported?

## Must Check

There are two bugs discovered from testing (so far):

* the state -> current_state bug we discussed on mattermost
* simulator seems to just break if epoch is not set to 1

Right now, there are `skip` decorators on thes tests that reveal these issues. You can remove that decorator and run pytest to see the errors. 

In addition to those, there is a mysterious deviding by 2 in `simulate/__init__.py` that I think should at least be explained.

All the other comments are more like suggestions optional to address.


