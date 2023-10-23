# TROT

This is a Python implementation of Tsallis Regularized Optimal Transport (TROT) for ecological inference, following

> Boris Muzellec, Richard Nock, Giorgio Patrini, Frank Nielsen. Tsallis Regularized Optimal Transport and Ecological Inference.  	[arXiv:1609.04495](https://arxiv.org/pdf/1609.04495v1.pdf)

It contains both scripts implementing algorithms for solving TROT, and notebooks which reproduce the ecological inference pipeline from the article.


# Dependencies

```
numpy, scipy, pickle
```

# Usage

To run the Ecological Inference notebook, you will first want to download the Florida dataset. It may be accessed [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SVY5VF).

It should then be stored in the root folder of the repo.

You can then run `Notebooks/Ecological/Inference.ipynb` for a reproduction of the article's ecological inference pipeline, and `Notebooks/Tsallis/Plots.ipynb` for a visualization of the impact of parameter $q$ and $\lambda$.

The code under `Trot/` contains the basics for building a TROT-based application.
