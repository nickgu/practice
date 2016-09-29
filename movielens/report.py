

import math
import sklearn.metrics as metrics
from simple_svd import inverse_ratio

pred = map(lambda l:float(l.strip()), file('pred.txt').readlines())
label = map(lambda l:float(l.split(' ')[0]), file('data/svdf_test.txt').readlines())

print 'MAE=%s' % metrics.mean_absolute_error(label, pred)
print 'MSE=%s' % metrics.mean_squared_error(label, pred)
print 'RMSE=%s' % math.sqrt( metrics.mean_squared_error(label, pred) )

order = zip( pred, label )
ir, ic, tot = inverse_ratio( order )
print 'inverse_ratio = %.3f%%' % (ir)


# select top 5 diff.
idx_diff = [ (idx, abs(pred-label), label, pred) for idx, (pred, label) in enumerate(order) ]
for idx, diff, label, pred in sorted(idx_diff, key=lambda x:-x[1])[:10]:
    print idx, diff, label, pred




