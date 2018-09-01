#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 


def common_train(train_data, model, epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.02)

    dtype = torch.FloatTensor
    loader = torch.utils.data.DataLoader(zip(X, Y), shuffle=True, batch_size=150)

    epoch_num = 0
    best_hit = 0
    best_ps = ''
    while 1:
        print >> sys.stderr, 'input n epoch to run..'
        l=sys.stdin.readline()
        try:
            n = int(l)
        except:
            break

        for i in range(n):
            epoch_num += 1
            
            data_iter = iter(loader)
            run_loss = 0
            right = 0
            total = 0
            for t, (x, y) in enumerate(data_iter):
                inputs = Variable(x).type(dtype)
                labels = Variable(y).type(torch.LongTensor)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)

                # temp test precision
                # seems different max need keepdim..
                #pred = outputs.max(1, keepdim=True)[1]
                pred = outputs.max(1)[1]

                p = pred.eq(labels.view_as(pred)).sum()
                right += p.data[0]
                total += len(pred)
                precision = right * 100.0 / total

                loss = criterion(outputs, labels)
                run_loss += loss.data[0]
                loss.backward()
                optimizer.step()

            # print statistics
            ops = 'p=%.2f%% (%d/%d)' % (precision, right, total)
            ps = ops
            if right >= 145:
                ps = pydev.ColorString.green(ops)
            if right >= 147:
                ps = pydev.ColorString.cyan(ops)
            if right >= 148:
                ps = pydev.ColorString.red(ops)
            if right >= 149:
                ps = pydev.ColorString.yellow(ops)

            if right > best_hit:
                best_hit = right
                best_ps = ps
   
            print('[%d, %d] loss: %.5f %s curbest=%s' %
              (epoch_num, n, run_loss, ps, best_ps))
            if best_hit == 150:
                # we found the best answers!.
                break

        if best_hit == 150:
            # we found the best answers!.
            break

   

if __name__=='__main__':
    pass