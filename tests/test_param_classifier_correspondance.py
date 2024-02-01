from pkgnametbd.simulations import sconfig
from pkgnametbd.simulations import parameter_space


pspace = parameter_space.parameter_values(
        sconfig.ERRORS_PARAMETER_SPACE_BEGIN,
        sconfig.ERRORS_PARAMETER_SPACE_END,
        sconfig.ERRORS_PARAMETER_SPACE_NUM
)
ft_params = [{
    "A": (mean_a, sconfig.FEATURE_DIST_VARIANCE),
    "B": (mean_b, sconfig.FEATURE_DIST_VARIANCE)}
    for (mean_a, mean_b) in pspace
 ]
print(ft_params)