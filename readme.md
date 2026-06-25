# Imputation Library

Commonly available imputation tools lack support for categorical values, and sometimes do not work at all.
This project wants to add robust imputation benchmarks, with significant performance boost to other available tools.

__Installation:__
´´´ bash
pip install gouda-cheese
uv add gouda-cheese
´´´

__Usage:__
´´´ python
from gouda import KnnImputer

df: pd.DataFrame

imputation_model = KnnImputer().fit(df)
imputed = imputation_model.transform(df)
´´´

# Imputation Algorithms
 - KNN 
 - MICE
 - Mean/Mode (Simple)
