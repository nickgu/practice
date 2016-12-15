

import sys
import math
import sklearn.metrics as metrics
from simple_svd import inverse_ratio

# test for ml-100k.
#svd_test_file = 'data/svdf_test.txt'
#original_ratings = 'ml-100k/ua.test'

# test for ml-1m.
svd_test_file = 'data/svdf_test.txt'
original_ratings = 'ml-1m/test.dat'

test_file = file(svd_test_file)
if len(sys.argv) > 1:
    print >> sys.stderr, 'TEST = %s' % sys.argv[1]
    test_file = file(sys.argv[1])

pred = map(lambda l:float(l.strip()), file('pred.txt').readlines())
label = map(lambda l:float(l.split(' ')[0]), test_file.readlines())

print >> sys.stderr, 'MSE=%s' % metrics.mean_squared_error(label, pred)
print >> sys.stderr, 'MAE=%s' % metrics.mean_absolute_error(label, pred)
print >> sys.stderr, 'RMSE=%s' % math.sqrt( metrics.mean_squared_error(label, pred) )

order = zip( pred, label )
ir, ic, tot = inverse_ratio( order )
print >> sys.stderr, 'inverse_ratio = %.3f%%' % (ir * 100.)

# select top 5 diff.
print 
print >> sys.stderr, 'ERROR SAMPLE'

test_data = file(original_ratings).readlines()
idx_diff = [ (idx, abs(pred-label), label, pred) for idx, (pred, label) in enumerate(order) ]
for idx, diff, label, pred in sorted(idx_diff, key=lambda x:-x[1]):
    if diff < 3:
        continue

    print 'sample_id=%d, diff=%.3f, label=%d, pred=%.3f, data=[%s]' % (idx, diff, int(label), pred, test_data[idx].strip())


    



