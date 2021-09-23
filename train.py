import torch
import torch.nn.functional as F 
import wandb


def train_iter(args, model, optimz, data_load, val_loader, test_loader,loss_train, loss_val, loss_test):
    samples = len(data_load.dataset)
    val_acc = []
    test_acc = []
    best_accuracy = 0
    wandb.watch(model, log="None", log_freq=10)
    model.train()
    model.cuda()
    #model.clean_activation_buffers()
    optimz.zero_grad()
    step = 0
    for epoch in range(args.num_epochs):
        print(f"epoch: {epoch}")
        for i, (data, target) in enumerate(data_load):
            target = target.squeeze(dim=1)
            out = F.log_softmax(model(data.cuda()), dim=1)
            loss = F.nll_loss(out, target.cuda())
            loss.backward()
            optimz.step()
        
            optimz.zero_grad()
        #model.clean_activation_buffers()
            if i % 100 == 0:
                step += int(100 * 100 / len(data_load))
                wandb.log({"epoch": epoch, "loss": loss}, step=step)
                print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                      ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                      '{:6.4f}'.format(loss.item()))
                loss_train.append(loss.item())
        val_accuracy = evaluate_val(model, val_loader, loss_val)
        val_acc.append(val_accuracy)
        test_accuracy = evaluate_test(model, test_loader, loss_test)
        test_acc.append(test_accuracy)
        if test_acc[-1] > best_accuracy:
            best_accuracy = test_acc[-1]
            best_save_path = f'{args.save_dir}/{args.model_name}_{args.data}_batchsize{args.batch_size}_window{args.num_frames}_best_epoch.pt'
            torch.save(model.state_dict(), best_save_path)
    last_save_path = f'{args.save_dir}/{args.model_name}_{args.data}_batchsize{args.batch_size}_window{args.num_frames}_epoch_{args.num_epochs}.pt'
    torch.save(model.state_dict(), last_save_path)


def evaluate_test(model, data_load, loss_val):
    model.eval()
    
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    #model.clean_activation_buffers()
    with torch.no_grad():
        for data,target in data_load:
            target = target.squeeze(dim=1)
            output = F.log_softmax(model(data.cuda()), dim=1)
            loss = F.nll_loss(output, target.cuda(), reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            tloss += loss.item()
            csamp += pred.eq(target.cuda()).sum()
            #model.clean_activation_buffers()
    aloss = tloss / samples
    loss_val.append(aloss)
    accuracy = 100.0 * csamp / samples
    wandb.log({"test_accuracy": accuracy})
    wandb.log({"test_loss": aloss})
    #torch.onnx.export(model, images, "model.onnx")
    #wandb.save("model.onnx")
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')
    return accuracy
    

def evaluate_val(model, data_load, loss_val):
    model.eval()
    
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    #model.clean_activation_buffers()
    with torch.no_grad():
        for data,target in data_load:
            target = target.squeeze(dim=1)
            output = F.log_softmax(model(data.cuda()), dim=1)
            loss = F.nll_loss(output, target.cuda(), reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            tloss += loss.item()
            csamp += pred.eq(target.cuda()).sum()
            #model.clean_activation_buffers()
    aloss = tloss / samples
    loss_val.append(aloss)
    accuracy = 100.0 * csamp / samples

    wandb.log({"val_accuracy": accuracy})
    wandb.log({"val_loss": aloss})
    #torch.onnx.export(model, images, "model.onnx")
    #wandb.save("model.onnx")
    print('\nAverage val loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')
    return accuracy