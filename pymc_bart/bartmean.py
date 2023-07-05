#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from typing import List, Optional, Tuple, Union
import numpy.typing as npt
import numpy as np
from numba import njit
import pytensor.tensor as pt
from pymc.model import Model, modelcontext
from pymc.pytensorf import inputvars, join_nonshared_inputs, make_shared_replacements
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.step_methods.compound import Competence
from pytensor import config
from pytensor import function as pytensor_function
from pytensor.tensor.var import Variable

from pymc.distributions.distribution import Distribution, _moment
from pymc.logprob.abstract import _logprob
from pytensor.tensor.random.op import RandomVariable
from pymc.distributions.distribution import Continuous

class BARTmeanRV(RandomVariable):
    """Supporting RV containing just the leaf means, to allow SD to flow through and become fittable"""
    name: str = "BARTmean"
    ndim_supp = 0
    ndims_params = [0]
    dtype = "floatX"
    _print_name: Tuple[str, str] = ("BARTmean", "\\operatorname{BARTmean}"),

    @classmethod
    def rng_fn(cls, rng, size):
        if size is not None:
            return np.full(size, 0.0)
        else:
            return 0.0

bartmean = BARTmeanRV()

class BARTMean(Continuous):
    rv_op = bartmean

    def __new__(cls, *args, **kwargs):
        kwargs.setdefault("initval", "moment")
        Distribution.register(BARTmeanRV)
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def dist(cls, **kwargs):
        return super().dist([],**kwargs)

    def moment(rv, size):
        return pt.zeros(size)

    def logp(value):
        return pt.zeros_like(value)

PGBART_means = {}

class BARTMeanDummy(ArrayStepShared):
    """
    Particle Gibss BART sampling step.

    Parameters
    ----------
    vars: list
        List of value variables for sampler
    num_particles : tuple
        Number of particles. Defaults to 10
    batch : int or tuple
        Number of trees fitted per step. Defaults to  "auto", which is the 10% of the `m` trees
        during tuning and after tuning. If a tuple is passed the first element is the batch size
        during tuning and the second the batch size after tuning.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    """

    name = "bartmean-dummy"
    default_blocked = False
    generates_stats = True
    stats_dtypes = [{}]

    def __init__(
        self,
        vars = None,  # pylint: disable=redefined-builtin
        model: Optional[Model] = None,
        **kwargs
    ):
        model = modelcontext(model)
        initial_values = model.initial_point()
        if vars is None:
            vars = model.value_vars
        else:
            vars = [model.rvs_to_values.get(var, var) for var in vars]
            vars = inputvars(vars)

        self.bart_name = 'BART'#vars[0].name[3:]
        #self.bartmean = model.values_to_rvs[value_bart].owner.op
        print("BMNAME",self.bart_name)

        shared = make_shared_replacements(initial_values, vars, model)
        super().__init__(vars, shared, **kwargs)

    def astep(self, pt):
        if self.bart_name in PGBART_means:
            #print("PTS",pt.data)
            #print("PNV",PGBART_means[self.bart_name].sum_trees_mean.flatten())
            pt = PGBART_means[self.bart_name].sum_trees_mean#.flatten()
        return pt,[{}]

    @staticmethod
    def competence(var, has_grad):
        """PGBART is only suitable for BART distributions."""
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTmeanRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE
