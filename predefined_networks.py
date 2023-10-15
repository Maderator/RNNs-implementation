def one_layered_lstm(output_units, lstm_units=64, return_sequences=False):
    """Basic one layered LSTM network

    Args:
        output_units (int): number of units in the last dense layer
        lstm_units (int): number of units in LSTM layer
        return_sequences (bool): whether to output whole sequences or only last output out of the model

    Returns:
        dict, list(dict), list(dict), int
        returns a dictionary with parameters for output layer, list of rnn layers, list out dense layers, and residual block size
    """
    lstm_layers = [
        {
            "cell_type": "lstm",
            "cell_kwargs": {"units": lstm_units},
            "layer_kwargs": {"return_sequences": return_sequences},
            "bidirectional": False,
        }
    ]
    output_layer_params = {
        "units": output_units,
    }
    return output_layer_params, lstm_layers, [], None


def smyl_std_lstm(
    output_units,
    cell_type="lstm",
    lstm_units=50,
    dilation_base=2,
    residual_block_size=2,
    num_layers=4,
    return_sequences=False,
):
    """Standard LSTM architecture as defined by Smyl

    It consists of four layers of dillated lstm layers with residual connection between the input to third layer and the output of fourth layer.
    See A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting by Smyl S. https://www.sciencedirect.com/science/article/pii/S0169207019301153 for more information

    Args:
        output_units (int): number of units in the last dense layer
        cell_type (string): either "lstm" or "gru" (type of rnn layer)
        lstm_units (int): number of units in rnn layer
        dilation_base (int): each layer has a dilation rate which is computed as dilation_base**layer_number
        residual_block_size (int or None): Size of rnn block which has a residual connection between its input and output.
        num_layers (int): number of rnn layers
        return_sequences (bool): whether to output whole sequences or only last output out of the model

    Returns:
        dict, list(dict), list(dict), int
        returns a dictionary with parameters for output layer, list of rnn layers, list out dense layers, and residual block size
    """
    lstm_layers_params = []
    for i in range(num_layers):
        return_sequences = True
        if i == num_layers - 1:
            return_sequences = return_sequences
        layer_kwargs = {
            "cell_type": cell_type,
            "cell_kwargs": {"units": lstm_units},
            "layer_kwargs": {"return_sequences": return_sequences},
            "dilation_kwargs": {"dilation_rate": dilation_base**i},
        }
        lstm_layers_params.append(layer_kwargs)

    output_layer_params = {
        "units": output_units,
        "activation": None,  # linear adapter (adaptor)
    }
    return output_layer_params, lstm_layers_params, [], residual_block_size


def smyl_residual_lstm(
    output_units,
    cell_type="lstm",
    lstm_units=50,
    dilation_base=3,
    residual_block_size=None,
    num_layers=4,
    return_sequences=False,
):
    """LSTM architecture as defined by Smyl with residual connections (Kim et al. style)

    It consists of four layers of dillated lstm layers with residual connections between the input of each layer and the output of a cell state.
    See A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting by Smyl S. https://www.sciencedirect.com/science/article/pii/S0169207019301153 for more information.

    Args:
        output_units (int): number of units in the last dense layer
        cell_type (string): either "lstm" or "gru" (type of rnn layer)
        lstm_units (int): number of units in rnn layer
        dilation_base (int): each layer has a dilation rate which is computed as dilation_base**layer_number
        residual_block_size (int or None): Size of rnn block which has a residual connection between its input and output.
        num_layers (int): number of rnn layers
        return_sequences (bool): whether to output whole sequences or only last output out of the model

    Returns:
        dict, list(dict), list(dict), int
        returns a dictionary with parameters for output layer, list of rnn layers, list out dense layers, and residual block size
    """
    lstm_layers_params = []
    for i in range(num_layers):
        return_sequences = True
        if i == num_layers - 1:
            return_sequences = return_sequences
        layer_kwargs = {
            "cell_type": cell_type,
            "cell_kwargs": {"units": lstm_units, "residual_connection": True},
            "layer_kwargs": {"return_sequences": return_sequences},
            "dilation_kwargs": {"dilation_rate": dilation_base**i},
        }
        lstm_layers_params.append(layer_kwargs)

    output_layer_params = {
        "units": output_units,
        "activation": None,  # linear adapter (adaptor)
    }
    return output_layer_params, lstm_layers_params, [], residual_block_size


def smyl_attention_lstm(
    output_units,
    lstm_units=50,
    dilation_base=6,
    residual_block_size=None,
    num_layers=2,
    return_sequences=False,
):
    raise NotImplementedError()
    lstm_layers_params = []
    for i in range(num_layers):
        return_sequences = True
        if i == num_layers - 1:
            return_sequences = False
        layer_kwargs = {
            "cell_type": "lstm",
            "cell_kwargs": {
                "units": lstm_units
            },  # TODO set attention to True when implemented
            "layer_kwargs": {"return_sequences": return_sequences},
            "dilation_kwargs": {"dilation_rate": dilation_base**i},
        }
        lstm_layers_params.append(layer_kwargs)

    output_layer_params = {
        "units": output_units,
        "activation": None,  # linear adapter (adaptor)
    }
    return output_layer_params, lstm_layers_params, [], residual_block_size
