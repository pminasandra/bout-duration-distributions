import survival

def _markovised_seq_parallel_helper(markovised_sequence, count_markov,
                            species_, state, hazard_rate, add_bootstrapping):

    print(f"Working on Markovisation #{count_markov + 1}")
    survival_table_m = survival.compute_behavioural_inertia(
                            markovised_sequence,
                            species_,
                            state,
                            hazard_rate=hazard_rate
                        )
    if survival._is_invalid(survival_table_m):
        return "invalid"
    if add_bootstrapping:
        bootstrap_table_m = survival.bootstrap_and_analyse(
                                markovised_sequence,
                                species_,
                                state,
                                hazard_rate=hazard_rate
                            )
        upper_lim, lower_lim = survival._get_95_percent_cis(bootstrap_table_m)
        return survival_table_m, upper_lim, lower_lim

    return survival_table_m
