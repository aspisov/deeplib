import math
import deeplib

def _calculate_fan(tensor):
    dimensions = tensor.dim()
    if dimensions == 2:  # Dense layer
        fan_in, fan_out = tensor.shape
    elif dimensions > 2:  # Convolutional layer
        receptive_field_size = 1
        for dim in tensor.shape[2:]:
            receptive_field_size *= dim
        fan_in = tensor.shape[1] * receptive_field_size
        fan_out = tensor.shape[0] * receptive_field_size
    else:
        raise ValueError("Unsupported tensor shape")
    
    return fan_in, fan_out


def xavier_normal_(tensor):
    fan_in, fan_out = _calculate_fan(tensor)
    std = math.sqrt(2.0 / (fan_in + fan_out))
    with deeplib.no_grad():
        return tensor.normal_(0.0, std)

def kaiming_normal_(tensor, nonlinearity="relu"):
    gain = {
        "tah": 5.0 / 3,
        "relu": math.sqrt(2.0),
    }
    
    fan_in, _ = _calculate_fan(tensor)
    std = gain[nonlinearity] / math.sqrt(fan_in)
    with deeplib.no_grad():
        return tensor.normal_(0.0, std)
    
    