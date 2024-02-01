import pytest

from pkgnametbd.simulations.parameter_space import parameter_values


@pytest.mark.parametrize("error_num", [1, 14, 20])
def test_return_size_changes_with_errnum(error_num):
    param_space = [
        (x, y) for x, y in 
        parameter_values(10**-2, .99, error_num)
    ]
    assert len(param_space) == error_num
    assert all([x==-y for x, y in param_space])
    # max is 2.5758
    assert all([ -3 < x < 0 < y < 3 for x, y in param_space])