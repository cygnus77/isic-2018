import numpy as np

def validate(data_loader, net, criterion, measures, epoch):
    val_loss = 0.
    measurements = {k:0. for k in measures.keys()}
    for i, (inputs, labels) in enumerate(data_loader, 0):
        print("Validating epoch %d: batch # %d" % (epoch, i), end='\r')
        # map to gpu
        inputs, labels = inputs.cuda(), labels.cuda()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        for (k,mobj) in measures.items():
            m = mobj[0] # fn
            measurements[k] += m(outputs, labels).item()
        
        val_loss += loss.item()
    
    for k in measures.keys():
        measurements[k] = measurements[k] / len(data_loader)
    return val_loss / len(data_loader), measurements

def fit(net, train_loader, val_loader, criterion, optimizer, lrscheduler, measures, epoch, loss_vis):
    net.train(True)
    train_loss = 0.
    epoch_size = len(train_loader)
    losses=[]
    for i, (inputs, labels) in enumerate(train_loader, 0):
        print("Training epoch %d: batch # %d" % (epoch, i), end='\r')
        # map to gpu
        inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        losses.append(loss.item())
        if i % 30 == 0:
            loss_vis.plot_loss(np.mean(losses), (epoch_size * epoch) + i, 'train_loss')
            losses.clear()
            
            net.train(False)
            val_loss, measurements = validate(val_loader, net, criterion, measures, epoch)
            loss_vis.plot_loss(val_loss, (epoch_size * epoch) + i, 'val_loss')
            net.train(True)
            lrscheduler.step(val_loss)
            
            for k in measures.keys():
                measures[k][1].plot_loss(measurements[k], (epoch_size * epoch) + i, k)
        
    measurements['train_loss'] = train_loss / epoch_size
    measurements['val_loss'] = val_loss
    return measurements
