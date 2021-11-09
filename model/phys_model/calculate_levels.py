import numpy as np


def convert_max_into_delta(observed_level, predicted_max):
    """ The function converts the predicted level values (stage_max) to the target delta_stage_max variable

    :param observed_level: water level value that was known at the beginning of the forecast
    :param predicted_max: predicted values of levels for 7 days ahead
    :return delta_levels: level difference
    """
    shifted = predicted_max[:-1]
    new_arr = np.hstack([np.array(observed_level), shifted])

    delta_stage_max_predicted = predicted_max - new_arr
    return delta_stage_max_predicted
