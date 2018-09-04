#! /bin/sh

#movielens_path=./ml-100k/
movielens_path=./ml-1m/

#mkdir data
if [ "$1" == "no_dump" ]; then
    echo "No need to dump data"
else
    python dump_svdfeature_data.py $movielens_path
fi

mkdir models
rm -rf models/*

./svd_feature movielens_svdfeature.conf

model_id=`grep 'num_round=' movielens_svdfeature.conf | sed 's/.*=//g'`
echo "model_id=$model_id"

./svd_feature_infer movielens_svdfeature.conf start=0 end=$model_id > train.rmse.log
./svd_feature_infer movielens_svdfeature_test.conf start=0 end=$model_id > test.rmse.log

#python report.py data/svdf_train.txt

