MovieLens dataset train and test
===

100k Training.
---

### How to use svdfeature to train.

this scrip make train.in test.in in data/

> mkdir data

> python dump_svdfeature_data.py <movielens_path>

> mkdir models

> ./svd_feature movielens_svdfeature.conf

> \# run a infer and write result to pred.txt

> ./svd_feature_infer movielens_svdfeature.conf pred=500

> \# report a result of pred.txt

> python report.py

