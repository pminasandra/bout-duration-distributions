import pytest

import numpy as np
import pandas as pd

from pkgnametbd.simulations.agentpool import AgentPool
from pkgnametbd.simulations.agentpoolutils import recs_as_pd_dataframes

@pytest.mark.parametrize('prob_switching,unifdata,init_condition,expected_result',[
    (
        lambda size: .5,
        np.array([
            [.6, .6, .1, .1],
            [.6, .6, .1, .1],
        ]),
        np.array([-1, -1, -1, -1]),
        np.array([
            [-1, -1, -1, -1],
            [-1, -1, 1, 1],
            [-1, -1, -1, -1]

        ]),

    ),
    (
        lambda num_in_target_state: {
            4: .1,
            3: .2,
            2: .3,
            1: .4,
            0: .5
        }[num_in_target_state],
        np.array([
            # ps = .5
            [.6, .6, .1, .1],
            # ps = .3
            [.1, .4, .1, .4],
            # ps = .3
            [.4, .1, .1, .1],
            # ps = .4, .2
            [.5, .3, .5, .3]
        ]),
        np.array([-1, -1, -1, -1]),
        np.array([
            [-1, -1, -1, -1],
            # num = 0, ps = .5
            [-1, -1, 1, 1],
            # size = 2, ps = .3
            [1, -1, -1, 1],
            # size = 2, ps = .3
            [1, 1, 1, -1],
            # size = 1, ps = .3 and size=3 ps=.2
            [1, -1, 1, -1]

        ])

    )
])
def test_agent_pool(prob_switching, unifdata, init_condition, expected_result, mocker):

    t, num_agents = expected_result.shape
    u = unifdata.__iter__()
    mocker.patch('numpy.random.uniform', lambda size: next(u))

    pool = AgentPool(num_agents, prob_switching, init_condition)
    pool.run(t-1)

    np.testing.assert_array_equal(pool.data, expected_result)
    
@pytest.mark.parametrize('data,expected_states',[
    (
        np.array([
            [-1, -1, -1, -1],
            [-1, -1, 1, 1],
            [-1, -1, -1, -1]

        ]),
        ['AAA', 'AAA', 'ABA', 'ABA']
    ),
    (
        np.array([
            [-1, -1, -1, -1],
            [-1, -1, 1, 1],
            [1, -1, -1, 1],
            [1, 1, 1, -1],
            [1, -1, 1, -1]
        ]),
        ['AABBB', 'AAABA', 'ABABB', 'ABBAA']
    )
])
def test_recs_as_pd_dataframes(data, expected_states, mocker):
    mocker.patch('pkgnametbd.boutparsing.as_bouts', lambda x, species: x)
    expected_dfs =  [
        pd.DataFrame({
            'datetime': pd.to_datetime(range(len(expected_states[0])), unit='s'),
            'state': list(s)
        })
        for s in expected_states
    ]
    for actual, expected in zip(recs_as_pd_dataframes(data), expected_dfs):
        assert all(actual == expected)