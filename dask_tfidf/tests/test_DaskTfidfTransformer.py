import pytest
import dask.bag as db
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from dask_ml.feature_extraction.text import CountVectorizer

#typing import
from dask.array.core import Array as DaskArray


from ..dask_tf_idf import DaskTfidfTransformer

#from the dask.ml text testing cases
JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright"
    "the coke burger coke copyright",
    "the coke burger burger",
)

@pytest.fixture
def data() -> DaskArray:
    corpus = db.from_sequence(JUNK_FOOD_DOCS)
    cv = CountVectorizer()
    data = cv.fit_transform(corpus)
    data = data.persist()
    return data


@pytest.mark.parametrize("norm", [None, "l2","l1"])
@pytest.mark.parametrize("smooth_idf", [True, False])
@pytest.mark.parametrize("sublinear_tf", [True, False])
@pytest.mark.parametrize("use_idf", [True, False])
def test_tf_idf(data,norm,smooth_idf,sublinear_tf,use_idf):
    d_tf_idf =  DaskTfidfTransformer(norm=norm,smooth_idf=smooth_idf,sublinear_tf=sublinear_tf,use_idf=use_idf)
    d_tf_idf_result =  d_tf_idf.fit_transform(data).compute()


    data_np = data.compute()
    sk_tf_idf = TfidfTransformer(norm=norm,smooth_idf=smooth_idf,sublinear_tf=sublinear_tf,use_idf=use_idf)
    sk_tf_idf_result =  sk_tf_idf.fit_transform(data_np)

    np.testing.assert_array_almost_equal(d_tf_idf_result.todense().astype(np.float64),sk_tf_idf_result.todense().astype(np.float64))

