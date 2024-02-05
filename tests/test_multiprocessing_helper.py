import pytest
from unittest.mock import MagicMock

from collections import namedtuple
from copy import copy

import numpy as np
import pandas as pd

from pkgnametbd.simulations import _multiprocessing_helper_func

# This does not attempt to test:
# * simulations.mixed_exponentials.MixedExponentials
# * fitting.fit_all_states
# * fitting.choose_best_distribution
# Merely checks that the data passes in the way the 
# code suggests, and summary statistics are correctly recorded.

@pytest.mark.parametrize('kwargs,mockbouts', [
    ({'p': .5, 'expl0': .01, 'expl1': .001, 'count': 0, 'tgtlist': [1,2,3,4], 'num_sims': np.nan}, np.array([0, 0, 1, 1])),
    ({'p': .6, 'expl0': .002, 'expl1': .01, 'count': 3, 'tgtlist': [1,2,3,4], 'num_sims': np.nan}, np.array([1, 2, 3, 4, 5])),
    ({'p': .6, 'expl0': .002, 'expl1': .01, 'count': 88, 'tgtlist': [0]*100, 'num_sims': np.nan}, np.array([9, 3, 3, 3, 9, 3])),
])
def test_passthrough(kwargs, mockbouts, mocker):
    ## Prepare Mocks
    mocks = {
        'pkgnametbd.simulations.mixed_exponentials.MixedExponential': MagicMock(
           side_effect = lambda p_0, exp_lambda_0, exp_lambda_1: namedtuple('MockMixedExponential', 'generate_random')(
               generate_random=lambda size: mockbouts
           )
        ),
        'pkgnametbd.fitting.fits_to_all_states': MagicMock(
            side_effect = lambda bouts: {
                state: 'fit' + state
                for state in np.unique(bouts['state'])
            }
        ),
        'pkgnametbd.fitting.choose_best_distribution': MagicMock(
            side_effect = lambda fit, bouts: ('BestDist', 'throwaway')
        ), 
        'pkgnametbd.simulations.sconfig.NUM_BOUTS': len(mockbouts)
    }
    for name_fn, mock_fn in mocks.items():
        mocker.patch(
            name_fn,
            mock_fn
        )

    expected_tgtlist = copy(kwargs['tgtlist'])
    expected_tgtlist[kwargs['count']] = 'BestDist'
    ## Run function
    _multiprocessing_helper_func(**kwargs)
    
    # Assert calls to fits_all_states
    mocks['pkgnametbd.fitting.fits_to_all_states'].assert_called_once()
    assert all(mocks['pkgnametbd.fitting.fits_to_all_states'].call_args[0][0] == \
        pd.DataFrame.from_dict({
            'state': ["A"]*len(mockbouts),
            'duration': mockbouts   
        })
    )

    # Assert calls to choose_best_dist
    mocks['pkgnametbd.fitting.choose_best_distribution'].assert_called_once()
    assert mocks['pkgnametbd.fitting.choose_best_distribution'].call_args[0][0] == 'fitA'
    assert all(mocks['pkgnametbd.fitting.choose_best_distribution'].call_args[0][1] == \
        mockbouts
    )
    
    # Assert tgtlist (output) is recorded
    actual_tgtlist = kwargs['tgtlist']
    assert actual_tgtlist == expected_tgtlist