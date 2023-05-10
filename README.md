# dask_tfidf
A Dask native implementation of 'Term Frequency Inverse Document Frequency' for dask-ml and scikit-learn

Install
-------
>pip install git+https://github.com/pr38/dask_tfidf

This project simply includes a DaskTfidfTransformer class, which is more or less a dask equivalent for sklearn' TfidfTransformer.
It assumes a dask array of a counted tolkens, like the kind that dask_ml's CountVectorizer class creats.
DaskTfidfTransformer, has all the parameters/hyperparameters as sklearn' TfidfTransformer; namley 'norm', 'use_idf', 'smooth_idf' and 'sublinear_tf'.
DaskTfidfTransformer output should be nearly identically to the TfidfTransformer; there will be some very very slight floating point diffrences(see tests). I belive these diffrences are due to my use of the sparse libary's implementation of COO and dask's array, as opposed to sklearn's use of scipy's COO and numpy array.

I have also included a 'persist_idf_array' parameters, where the IDF array is presisted for faster transforming after fitting. As with all dask-ml workloads, I recommend presisting the input array before any computation. I would also recommend running "compute_chunk_sizes" on your dask arrays before running this class.



