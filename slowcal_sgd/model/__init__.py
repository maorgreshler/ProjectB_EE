from .logistic_regression import LogisticRegressionModel
from .simple_conv import SimpleConv
from .simple_conv_3_channel import SimpleConv3Chan

MODEL_REGISTRY = {
    'logistic_regression': LogisticRegressionModel,
    'simple_conv': SimpleConv,
    'simple_conv_3_chan': SimpleConv3Chan
}

