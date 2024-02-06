import pytest
from unittest.mock import MagicMock

from collections import namedtuple

import numpy as np

from pkgnametbd.simulations.mixed_exponentials import MixedExponential


mocks = {
    'numpy.random.uniform': MagicMock(
        side_effect = lambda size: np.arange(0, 1, 1.0/size),
    ),
    'powerlaw.Exponential': MagicMock(
        # Returns an object where .generate_random always returns lambda
        side_effect = lambda xmin, parameters, discrete: namedtuple('MockExponential', 'generate_random')(
            generate_random=lambda size: np.repeat(parameters[0], size)
        )
    )
}

@pytest.mark.parametrize('p_0', [.1, .5, .9])
#@pytest.mark.parametrize('p_0', [.5])
@pytest.mark.parametrize('lambda_0', [1, 3, 10])
@pytest.mark.parametrize('lambda_1', [.1, .3, .8])
@pytest.mark.parametrize('size', [10, 100, 1000])
def test_dist_params_used(p_0, lambda_0, lambda_1, size, mocker):
    for name_fn, mock_fn in mocks.items():
        mocker.patch(
            name_fn,
            mock_fn
        )
    
    me = MixedExponential(p_0, lambda_0, lambda_1)
    r = me.generate_random(size)
    assert len(r) == size
    assert (r==lambda_0).sum() == pytest.approx(size*p_0, abs=2)
    assert (r==lambda_1).sum() == pytest.approx(size*(1-p_0), abs=2)