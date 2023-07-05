# pylint: disable=unused-argument
# pylint: disable=arguments-differ
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

from multiprocessing import Manager
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from pandas import DataFrame, Series
from pymc.distributions.distribution import Distribution, _moment
from pymc.logprob.abstract import _logprob
from pytensor.tensor.random.op import RandomVariable

from .tree import Tree
from .utils import TensorLike, _sample_posterior
from .split_rules import SplitRule

__all__ = ["BART"]

class BARTRV(RandomVariable):
    """Base class for BART."""

    name: str = "BART"
    ndim_supp = 1
    ndims_params: List[int] = [2, 1, 0, 0, 0, 1]
    dtype: str = "floatX"
    _print_name: Tuple[str, str] = ("BART", "\\operatorname{BART}")
    all_trees = List[List[List[Tree]]]

    def _supp_shape_from_params(self, dist_params, rep_param_idx=1, param_shapes=None):
        return (2,)+dist_params[0].eval().shape[:1]

    @classmethod
    def rng_fn(
        cls, rng=None, X=None, sd=None, m=None, alpha=None, beta=None, split_prior=None, Y=None, size=None
    ):
        if not cls.all_trees:
            if size is not None:
                return np.full((size[0], 2, cls.Y.shape[0]), cls.initial_mean)
            else:
                return np.full( (cls.Y.shape[0]), 2, cls.initial_mean)
        else:
            if size is not None:
                shape = size[0]
            else:
                shape = 1
            res = _sample_posterior(cls.all_trees, cls.X, rng=rng, shape=shape).squeeze().T[:,None,:]
            return np.concatenate( [ res, np.zeros_like(res) ], axis=1)
            



#bart = BARTRV() # Does not seem necessary any more


class BART(Distribution):
    r"""
    Bayesian Additive Regression Tree distribution.

    Distribution representing a sum over trees

    Parameters
    ----------
    X : TensorLike
        The covariate matrix.
    Y : TensorLike
        The response vector.
    m : int
        Number of trees.
    response : str
        How the leaf_node values are computed. Available options are ``constant``, ``linear`` or
        ``mix``. Defaults to ``constant``.
    alpha : float
        Controls the prior probability over the depth of the trees.
        Should be in the (0, 1) interval.
    beta : float
        Controls the prior probability over the number of leaves of the trees.
        Should be positive.
    split_prior : Optional[List[float]], default None.
        Each element of split_prior should be in the [0, 1] interval and the elements should sum to
        1. Otherwise they will be normalized.
        Defaults to 0, i.e. all covariates have the same prior probability to be selected.
    split_rules : Optional[SplitRule], default None
        List of SplitRule objects, one per column in input data.
        Allows using different split rules for different columns. Default is ContinuousSplitRule.
        Other options are OneHotSplitRule and SubsetSplitRule, both meant for categorical variables.
    shape: : Optional[Tuple], default None
        Specify the output shape. If shape is different from (len(X)) (the default), train a
        separate tree for each value in other dimensions.
    separate_trees : Optional[bool], default False
        When training multiple trees (by setting a shape parameter), the default behavior is to
        learn a joint tree structure and only have different leaf values for each.
        This flag forces a fully separate tree structure to be trained instead.
        This is unnecessary in many cases and is considerably slower, multiplying
        run-time roughly by number of dimensions.

    Notes
    -----
    The parameters ``alpha`` and ``beta`` parametrize the probability that a node at
    depth :math:`d \: (= 0, 1, 2,...)` is non-terminal, given by :math:`\alpha(1 + d)^{-\beta}`.
    The default values are :math:`\alpha = 0.95` and :math:`\beta = 2`.
    """

    def __new__(
        cls,
        name: str,
        X: TensorLike,
        m: int = 50,
        alpha: float = 0.95,
        beta: float = 2.0,
        response: str = "constant",
        split_prior: Optional[List[float]] = None,
        split_rules: Optional[SplitRule] = None,
        separate_trees: Optional[bool] = False,
        Y: Optional[TensorLike] = None,
        initial_mean: Optional[float] = None,
        sd: Optional[TensorLike] = None,
        **kwargs,
    ):
        manager = Manager()
        cls.all_trees = manager.list()

        X = preprocess_tensor(X)

        if Y is not None:
            Y = preprocess_tensor(Y)
        elif initial_mean is None or sd is None:
            raise Exception('Either Y or both sd and initial_mean need to be specified')

        if initial_mean is None:
            initial_mean = Y.mean()

        if sd is None:
            y_unique = np.unique(Y)
            is_bernoulli = (y_unique.size == 2 and np.all(y_unique == [0, 1]))
            sd = 3 if is_bernoulli else Y.std()

        if split_prior is None:
            split_prior = []

        out_shape = kwargs['shape'] if 'shape' in kwargs else len(X)

        bart_op = type(
            f"BART_{name}",
            (BARTRV,),
            dict(
                name="BART",
                all_trees=cls.all_trees,
                inplace=False,
                initial_mean=initial_mean,
                X=X,
                m=m,
                response=response,
                alpha=alpha,
                beta=beta,
                split_prior=split_prior,
                split_rules=split_rules,
                separate_trees=separate_trees,
                sd=sd,
                Y=Y
            ),
        )()

        Distribution.register(BARTRV)

        @_moment.register(BARTRV)
        def get_moment(rv, size, *rv_inputs):
            return cls.get_moment(rv, size, *rv_inputs)

        cls.rv_op = bart_op
        params = [X, sd, m, alpha, beta, split_prior, Y]
        return super().__new__(cls, name, *params, **kwargs)[:,0]

    @classmethod
    def dist(cls, *params, **kwargs):
        return super().dist(params, **kwargs)

    def logp(value, X, sigma, *inputs):
        """Calculate log probability.

        Parameters
        ----------
        x: numeric, TensorVariable
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        
        # Normal distribution 
        return -0.5 * pt.pow((value[:,1,:]-value[:,0,:]) / sigma, 2) - pt.log(pt.sqrt(2.0 * np.pi)) - pt.log(sigma)
        #return -0.5 * pt.pow(value[:,1,:], 2) - pt.log(pt.sqrt(2.0 * np.pi)) # Unit normal
        #return pt.zeros_like(value)

    @classmethod
    def get_moment(cls, rv, size, *rv_inputs):
        mean = pt.fill(size, rv.initial_mean)
        return mean


def preprocess_tensor(
    X: TensorLike
) -> npt.NDArray[np.float_]:
    if isinstance(X, (Series, DataFrame)):
        X = X.to_numpy()
    X = X.astype(float)
    return X


@_logprob.register(BARTRV)
def logp(op, value_var, *dist_params, **kwargs):
    _dist_params = dist_params[3:]
    value_var = value_var[0]
    return BART.logp(value_var, *_dist_params)  # pylint: disable=no-value-for-parameter
