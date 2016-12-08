#! /bin/sh

movielens_path=~/lab/datasets/movielens/100k/ml-100k/

#mkdir data
#python dump_svdfeature_data.py $movielens_path

mkdir models
rm -rf models/*

./svd_feature movielens_svdfeature.conf

model_id=`grep 'num_round=' movielens_svdfeature.conf | sed 's/.*=//g'`
echo "model_id=$model_id"

./svd_feature_infer movielens_svdfeature.conf pred=$model_id
python report.py

