import pytest

import numpy as np

from pkgnametbd.simulations.simulator import Simulator

class MockDist:
    def __init__(self, fn):
        self.generate_random = fn

cap_letters = map(chr, range(65, 91))

# SO#38065898
def np_remove_repeated_values(a):
    return np.concatenate((
            a[0:1],
            a[1:][a[1:] != a[0:len(a) - 1]]
    ))

def test_helper():
    h = np_remove_repeated_values
    a = np.array
    assert all(h(a([1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4])) == a([1, 2, 3, 4]))
    assert all(h(a([1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4])) == a([1, 2, 3, 4]))
    assert all(h(a([1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2])) == a([1, 2, 1, 2]))



# Check that Simulator runs without error
# with one, two, many bd_dist and ft_params
@pytest.mark.parametrize('n_ft_params', [1, 2, 5])
def test_varying_feature_count_doesnt_break(n_ft_params, mocker):
    epoch = 1
    nbouts = 4 # hit each state twice
    
    # No error, all error functions return 0
    mocker.patch(
        'numpy.random.normal',
        lambda mu, sigma, n:
            np.repeat(0, n)
    )

    simulator = Simulator(
        bd_distributions={
            'state1': MockDist(lambda n: np.repeat(1, n)),
            'state2': MockDist(lambda n: np.repeat(1, n))
        },
        # number of features set by input
        # feature params don't have any impact because
        # we are mocking np.random.normal
        ft_params=[
            {
                'state1': (0, 1),
                'state2': (0, 1)
            }
            for i in range(n_ft_params)
        ],
        epoch=epoch
    )

    simulator.run(nbouts)
    records = simulator.records

    # because each bout is length 1, total time sum(bout time) = nbouts
    # that time is broken up by epoch
    assert records['datetime'].tolist() == [i for i in range(0, nbouts, epoch)]

    # expect records to have column for each feature plus date and state
    assert len(records.columns) == n_ft_params + 2

    # expect the set of output states is the set of input states. 
    # this won't universally be true if e.g. if bouts are longer than 
    assert set(records['state'].tolist()) == set(['state1', 'state2'])

def test_multiple_false_path(mocker):
    mocker.patch('numpy.random.normal', lambda mu, sigma, n: np.repeat(0, n))

    simulator = Simulator(
        bd_distributions={
            'state1': MockDist(lambda n: np.repeat(1, n)),
            'state2': MockDist(lambda n: np.repeat(1, n))
        },
        ft_params= {'state1': (0, 1), 'state2': (0, 1)},
        epoch=1
    )
    simulator.run(4)


# Check that lenght of dist is correctly translated to bouts
# bd_dist : 1, 2, 3, 4
# Expected final sequence:
#   Ax1, Bx2, Ax3, Bx4
# Total = 10
def test_conversion_boutl_to_state(mocker):
    # no errors to simplify
    mocker.patch('numpy.random.normal', lambda mu, sigma, n: np.repeat(0, n))

    dist = MockDist(lambda n: np.array([1, 2, 3, 4]))
    simulator = Simulator(
        bd_distributions={'A': dist, 'B': dist},
        ft_params= {'A': (0, 1), 'B': (0, 1)},
        epoch=1
    )
    simulator.run(4)
    assert simulator.records['state'].tolist() == \
        ['A'] + ['B'] * 2 + ['A'] * 3 + ['B'] * 4 


# Check that bd_dist.generate_random is applied
# and, multiple values are used, correctly translated to bouts

# bd_dist A: 1, 2, 3, 4
# bd_dist B: 10, 11, 12, 13
# Expected final sequence: Ax1, Bx11, Ax3, Bx13
# Total = 28
def test_bd_dist_is_applied(mocker):
    # no errors to simplify
    mocker.patch('numpy.random.normal', lambda mu, sigma, n: np.repeat(0, n))

    simulator = Simulator(
        bd_distributions={
            'A': MockDist(lambda n: np.array([
                1, 2, 3, 4
            ])),
            'B': MockDist(lambda n: np.array([
                10, 11, 12, 13
            ]))
        },
        ft_params= {'A': (0, 1), 'B': (0, 1)},
        epoch=1
    )
    simulator.run(4)
    assert simulator.records['state'].to_list() == \
        ['A'] + ['B'] * 11 + ['A'] * 3 + ['B'] * 13


# check that n_bouts is respected
@pytest.mark.parametrize('num_bouts', range(0, 25, 4))
def test_num_bouts_respected(num_bouts, mocker):
    # no errors to simplify
    mocker.patch('numpy.random.normal', lambda mu, sigma, n: np.repeat(0, n))

    dist = MockDist(lambda n: np.array([1, 2, 3, 4]*int((n+4)/4)))
    simulator = Simulator(
        bd_distributions={'A': dist, 'B': dist},
        ft_params= {'A': (0, 1), 'B': (0, 1)},
        epoch=1
    )
    simulator.run(num_bouts)
    assert len(np_remove_repeated_values(simulator.records['state'].to_numpy())) == num_bouts

# test max rec time is respected
@pytest.mark.parametrize('t', [0, 10, 100])
def test_max_rec_time_respected(t, mocker):
    mocker.patch('numpy.random.normal', lambda mu, sigma, n: np.repeat(0, n))
    mocker.patch('pkgnametbd.simulations.sconfig.MAX_REC_TIME', t)

    dist = MockDist(lambda n: np.array([1, 2, 3, 4]*t))
    simulator = Simulator(
        bd_distributions={'A': dist, 'B': dist},
        ft_params= {'A': (0, 1), 'B': (0, 1)},
        epoch = 1
    )
    simulator.run(200)
    assert len(simulator.records) == t

# check that
#    * np.random.normal is used and values are not reused
#    * appropriate means and sd are applied to each state/feature vector combo
def test_ft_params_used(mocker):
    
    def mock_rnorm(mu, sigma, n):
        # returns a 4 digit number hinting at inputs
        # 1000s digit represents state mu (0-9)
        # 100s digit represents sigma (0-99)
        # based on feature generation below, each mu+sigma=feature number
        # Last 2 digits (10s, 1s) represent the times/observation number
        assert mu < 10
        assert sigma < 10
        assert n < 100
        return np.repeat(mu*1000, n) \
            + np.repeat(sigma*100, n) \
            + np.arange(0, n)

    mocker.patch('numpy.random.normal', mock_rnorm)

    dist = MockDist(lambda n: np.repeat(2, n))
    simulator = Simulator(
        # each bout is length 2
        bd_distributions={'A': dist, 'B': dist},
        ft_params= [
            # A vs B different values
            {'A': (1, 1), 'B': (2, 2)},
            # Same mean, different SD
            {'A': (1, 3), 'B': (1, 4)}
        ],
        epoch = 1
    )
    f0_expected = np.array([
        # A
        1100, 1101,
        # B
        2200, 2201,
        # A
        1100, 1101,
        # B
        2200, 2201,
        # A
        1100, 1101
    ])
    f1_expected = np.array([
        # A
        1300, 1301,
        # B
        1400, 1401,
        # A
        1300, 1301,
        # B
        1400, 1401,
        # A
        1300, 1301
    ])

    simulator.run(5)
    records = simulator.records
    assert all(records['feature0'].to_numpy() == f0_expected)
    assert all(records['feature1'].to_numpy() == f1_expected)

# check that epoch is respected and sequence is calcualted accurately
# from bout length