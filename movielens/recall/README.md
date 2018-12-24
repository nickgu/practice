# Movielens Data Train/Test Using As Binary

## Data Construction

### How to generate data?
use the ../dataset/movielens tools to generate data.

```
ml_binary_reader.py rating.csv output_dir
```

## File Description


### utils.py
To read train/test/valid data, and to measure the algorithm:
- readfile(): read one file(train/test/valid), test\_num is set to read data count.
- readdata(): read all data from directory(train/test/valid).
- measure(): test the algorithm. Give a predict function to get the answer.
    - predictor's imp: pred(uid, readitems): and return the answer list by item\_id list.

### ground\_truth.py

### ann.py

### dnn.py
