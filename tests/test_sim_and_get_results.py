import pytest
from unittest.mock import MagicMock

from collections import namedtuple
import numpy as np
import pandas as pd

from pkgnametbd.simulations import _simulate_and_get_results

# This does not attempt to test:
# * simulations.simulator.Simulator
# * boutparsing.as_bouts
# * fitting.fit_all_states
# * fitting.choose_best_distribution
# Merely checks that the data passes in the way the 
# code suggests, and summary statistics are correctly recorded.

# Returns a function that returns a named tuple
# that can stand in for the Simulator object.
# the run method of this object does nothing.
# the records attribute is whatever was passed into
# the parent function.
def build_mock_simulator(records):
    def MockSimulator(bd_distributions, ft_params, epoch):
        return namedtuple('MockSimulator', 'run num_features records')(
            run = lambda nbouts: None,
            num_features = len(records.columns) - 2,
            records = records
        )
    return MockSimulator

def mock_pred_dist(alpha):
    return namedtuple('MockPredDist', 'alpha')(alpha = alpha)

def to_float(letter_seq):
    # A is negative, B is positive
    return [
        {'A': -5, 'B': 5}[l]
        for l in letter_seq
    ]

def to_letter(float_seq):
    return [
        {True: 'A', False: 'B'}[l > 0]
        for l in float_seq
    ]

def lookup(key, value):
    return {
        f_config[key]: f_config[value]
        for f_config in FEATURE_CONFIG.values()
    }


FEATURE_CONFIG = {
    'feature0': {
        'charseq':'AAAAAAAAAABBBBBBBBBBAAAAAAAAAABBBBBBBBBB',
        'bouts': pd.DataFrame.from_dict({
            'state' : ['A', 'B'] * 2,
            'duration' : [10, 10, 10, 10],
        }),
        'fits': {
            'A': 'feature0fitA',
            'B': 'feature0fitB'
        },
        "dist": {
            'A': ('Truncated_Power_Law', mock_pred_dist(0)),
            'B': ('Power_Law', mock_pred_dist(0))
        },
        'fit_results': 1,
        'fit_results_spec': 1
    },
    'feature1': {
        'charseq': 'BAAAAAAAAABBBBBABBBBAAAAAAAAAABBBBBBBAAB',
        'bouts': pd.DataFrame.from_dict({
            'state' : ['B'] + ['A', 'B'] * 4,
            'duration' : [1, 9, 5, 1, 4, 10, 7, 2, 1],
        }),
        'fits': {
            'A': 'feature1fitA',
            'B': 'feature1fitB'
        },
        'dist': {
            'A': ('Power_Law', mock_pred_dist(4)),
            'B': ('not_heavy_tail', mock_pred_dist(0))
        },
        'fit_results': .5,
        'fit_results_spec': 0
    },
    'feature2': {
        'charseq': 'ABABABABABABABABABABABABABABABABABABABAB',
        'bouts': pd.DataFrame.from_dict({
            'state' : ['A', 'B'] * 10,
            'duration' : [1] * 20,
        }),
        'fits': {
            'A': 'feature2fitA',
            'B': 'feature2fitB'
        },
        'dist': {
            'A': ('not_heavy_tail', mock_pred_dist(0)),
            'B': ('not_heavy_tail', mock_pred_dist(0))
        },
        'fit_results': 0,
        'fit_results_spec': 0
    }
}
def test_passthrough_all_features(mocker):
    ## Prepare Mocks
    records = pd.DataFrame.from_dict({
        'state': list(
            'A'*10 + 'B'*10 + 'A'*10 + 'B'*10
        ),
        'datetime': pd.to_datetime(range(40), unit='s'),
        # perfect
        'feature0': to_float(list(FEATURE_CONFIG['feature0']['charseq'])),
        # wrong 10% of the time
        'feature1': to_float(list(FEATURE_CONFIG['feature1']['charseq'])),
        # arbitrary
        'feature2': to_float(list(FEATURE_CONFIG['feature2']['charseq'])),
    })
    mocks = {
        # note this is has to be .simulations.Simulator
        # not simulations.simulator.Simulator
        # because of relative imports?
        'pkgnametbd.simulations.Simulator': build_mock_simulator(records),
        'pkgnametbd.boutparsing.as_bouts': MagicMock(
            side_effect = lambda dataframe, species, randomize = False: lookup(
                'charseq',
                'bouts'
            )[''.join(dataframe['state'])]
        ),
        'pkgnametbd.fitting.fits_to_all_states': MagicMock(
            side_effect = lambda bouts: FEATURE_CONFIG[{
                10: 'feature0',
                9: 'feature1',
                1: 'feature2'
            # use the 2nd element to determine which feature has been called
            }[bouts['duration'][1]]]['fits']
        ),
        'pkgnametbd.fitting.choose_best_distribution': MagicMock(
            side_effect = lambda fit, bouts: FEATURE_CONFIG[fit[0:8]]['dist'][fit[11]]
        ),
        'pkgnametbd.config.xmin': 0
    }
    for name_fn, mock_fn in mocks.items():
        mocker.patch(
            name_fn,
            mock_fn
        )
    fit_results = list()
    fit_results_spec = list()

    ## Run function
    _simulate_and_get_results(
        # this is just a label
        sim_count=1,
        # need to have the correct dimensions here
        ft_params=[
            (0, 0),
            (1, 1),
            (2, 2)
        ],
        # just passed to simulator which we're mocking
        bd_distributions=None,
        # simulator breaks for other values
        epoch=1,
        fit_results=fit_results,
        fit_results_spec=fit_results_spec
    )
    ## Assert Expectations
    # TODO: run assertions
    '''
    as_bout_cal = mocks['pkgnametbd.boutparsing.as_bouts'].call_args_list
    expected_state_seqs = set(
        [
            v['charseq'] for v in FEATURE_CONFIG.values()
        ]
    )
    actual_state_seqs = set()
    for argd in as_bout_cal:
        assert 'dataframe' in argd
        assert 'state' in argd['dataframe']
        actual_state_seqs.add(''.join(argd['dataframe']['state'].tolist()))

    assert expected_state_seqs == actual_state_seqs
    '''
    assert len(fit_results) == 1
    assert len(fit_results_spec) == 1
    assert fit_results[0] == [v['fit_results'] for v in FEATURE_CONFIG.values()]
    assert fit_results_spec[0] == [v['fit_results_spec'] for v in FEATURE_CONFIG.values()]
    