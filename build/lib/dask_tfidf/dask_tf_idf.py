import dask.array as da
import numpy as np
import sparse
import math

from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.exceptions import NotFittedError

# typing imports
from typing import Optional, Any, Literal
from dask.array.core import Array as DaskArray


def _dask_handle_zeros_in_scale(scale: DaskArray) -> DaskArray:
    """
    A dask native vertion of sklearn's 'preprocessing._data._handle_zeros_in_scale'.
    The'copy' and precalculated 'mask' functionality has been stripped.
    Used to replace 0's and very small floating numbers with 1's, for the context of scaleing.
    Assumes a 1d array.
    """
    constant_mask = scale < 10 * np.finfo(scale.dtype).eps

    scale[constant_mask] = 1.0

    return scale


def _dask_normalize(X: DaskArray, norm: Literal["l2", "l1", "max"] = "l2") -> DaskArray:
    """
    A quick dask native copy of sklean's 'preprocessing._data.pynormalize'.
    All the scipy.parse functionality has been striped off.
    """

    if norm not in ("l1", "l2", "max"):
        raise ValueError("'%s' is not a supported norm" % norm)

    if norm == "l1":
        norms = da.abs(X).sum(axis=1)
    elif norm == "l2":
        norms = da.sqrt(da.square(X).sum(axis=1))
    elif norm == "max":
        norms = da.max(da.abs(X), axis=1)

    norms = _dask_handle_zeros_in_scale(norms)
    X = X / norms[:, np.newaxis]

    return X


class DaskTfidfTransformer(
    OneToOneFeatureMixin, TransformerMixin, BaseEstimator, auto_wrap_output_keys=None
):

    idf_: DaskArray
    norm: Optional[Literal["l2", "l1"]]
    smooth_idf: bool
    sublinear_tf: bool
    is_fitted: bool
    persist_idf_array: bool

    def __init__(
        self,
        *,
        norm: Optional[Literal["l2", "l1"]] = "l2",
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
        persist_idf_array: bool = True,
    ):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.persist_idf_array = persist_idf_array

        self.is_fitted = False

    def fit(self, X: DaskArray, y: Any = None):

        if self.use_idf:

            X = X.map_blocks(lambda a: sparse.as_coo(a).astype(np.float64))

            if math.isnan(
                X.shape[0]
            ):  # if number is rows is not currently know, get it
                X.compute_chunk_sizes()

            n_samples = X.shape[0]

            df = da.count_nonzero(X, axis=0)

            if self.smooth_idf:
                n_samples = n_samples + 1
                df = df + 1

            self.idf_ = da.log(n_samples / df) + 1

            if self.persist_idf_array:
                self.idf_ = self.idf_.persist()

        self.is_fitted = True

        return self

    def transform(self, X: DaskArray) -> DaskArray:

        if not self.is_fitted:
            NotFittedError("The Dask TF-IDF Transformer is not fitted")

        X = X.map_blocks(lambda a: sparse.as_coo(a).astype(np.float64))

        if self.sublinear_tf:
            X = da.where(X != 0,da.log(X) + 1,X)

        if self.use_idf:
            tf_idf = X * self.idf_
        else:
            tf_idf = X

        if self.norm:
            tf_idf = _dask_normalize(tf_idf, norm=self.norm)

        return tf_idf
