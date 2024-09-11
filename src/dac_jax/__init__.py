__version__ = "1.1.0"

__author__ = """David Braun"""
__email__ = "braun@ccrma.stanford.edu"

from dac_jax import nn
from dac_jax import model
from dac_jax import utils
from dac_jax.utils import load_model, load_encodec_model
from dac_jax.model import DACFile
from dac_jax.model import DAC
from dac_jax.model import EncodecModel
from dac_jax.nn.quantize import QuantizedResult
