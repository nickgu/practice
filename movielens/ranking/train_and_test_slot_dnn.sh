#!  /bin/sh


# join data.
# join train data.
python join_data.py --op:join -f data/train  -m data/ml-20m -s temp/slot.info -o temp/ins.txt -c temp/coder.out
# join test data.
python join_data.py --op:join -f data/test  -m data/ml-20m -s temp/slot.test.info -o temp/ins.test.txt -c temp/coder.test.out

# train model
python train_slot_dnn.py  -f temp/ins.txt  -o temp/slot_dnn.pkl -s temp/slot.info  -emb 8

# test model.
python model_tester.py --op:slot_dnn -emb 8 -s temp/slot.info  -m temp/slot_dnn.pkl -test temp/ins.test.txt

# test data from train.
head -n 500000 temp/ins.txt > temp/ins.test.fromtrain.txt
python model_tester.py --op:slot_dnn -emb 8 -s temp/slot.info  -m temp/slot_dnn.pkl -test temp/ins.test.fromtrain.txt

