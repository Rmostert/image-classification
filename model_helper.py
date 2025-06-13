def test_validation_stats(model,dataloader,criterion,device):
    
    '''
    Calculates the validation accuracy rate and loss 
    Arguments:
        model: A Pytorch model 
        dataloader: A Pytorch object returned by torch.utils.data.DataLoader
        criterion: The statistic that will be used to evaluate the loss
        device: Whether the scoring should be done on the CPU or GPU
    ''' 
    
    import torch
    
    loss = 0
    correct = 0
    total = 0


    for i, (valid_images, valid_labels) in enumerate(dataloader):

        valid_images, valid_labels = valid_images.to(device), valid_labels.to(device)
        outputs = model.forward(valid_images)
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == valid_labels).sum().item()
        total += valid_labels.size(0)
        accuracy = correct/total
        loss += criterion(outputs, valid_labels).item()

    return accuracy,loss


def train_model(model, trainloader, validloader, optimizer,criterion,epochs, device,print_every=40):
    
    '''
    Trains a Pytorch model
    Arguments:
        model:A Pytorch model 
        trainloader: The training data. Should be a Pytorch object returned by torch.utils.data.DataLoader
        validloader: The validation data. Should be a Pytorch object returned by torch.utils.data.DataLoader
        optimizer: the optimizer that should be used
        criterion: The statistic that will be used to evaluate the loss
        epochs: The number of epochs
        device: Whether the model training should be done on the CPU or GPU
        print_every: After how many steps the los should be printed
    '''
    
    import torch

    model = model.to(device)

    steps = 0

    for e in range(epochs):
        running_loss = 0

        for i, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                model.eval()

                with torch.no_grad():
                    valid_accuracy,valid_loss = test_validation_stats(model,validloader,criterion,device)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(valid_loss),
                      "Validation Accuracy: {:.2f}".format(valid_accuracy*100))



                running_loss = 0
                model.train()
    return model

def initialise_model(architecture,hidden_units,output_size,lr):
    
    '''
    Initialises a Pytorch model by making use of a transfer solution approach
    Arguments:
        architecture: The transfer solution architecture. Currently olny support VGG16, Densenet121 and Resnet50
        hidden_units: The number of hiden units in the network
        output_size: The output size of the netword, i.e. number of categories to predict
        lr: The learning rate
        
    '''

    import torchvision.models as models
    from torch import nn
    from collections import OrderedDict
    from torch import optim



    if architecture == 'vgg16':
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(0.5)),
                              ('fc2', nn.Linear(hidden_units, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        for param in model.parameters():
            param.requires_grad = False
        
        model.classifier = classifier

    elif architecture == 'densenet121':
        classifier = nn.Sequential(OrderedDict([
                                      ('fc1', nn.Linear(1024, hidden_units)),
                                      ('relu', nn.ReLU()),
                                      ('dropout1', nn.Dropout(0.5)),
                                      ('fc2', nn.Linear(hidden_units, output_size)),
                                      ('output', nn.LogSoftmax(dim=1))
                                      ]))
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        for param in model.parameters():
            param.requires_grad = False
        
        model.classifier = classifier

    elif architecture == 'resnet50':
        classifier = nn.Sequential(OrderedDict([
                                      ('fc1', nn.Linear(2048, hidden_units)),
                                      ('relu', nn.ReLU()),
                                      ('dropout1', nn.Dropout(0.5)),
                                      ('fc2', nn.Linear(hidden_units, output_size)),
                                      ('output', nn.LogSoftmax(dim=1))
                                      ]))
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = classifier
    

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)


    return model, criterion, optimizer


def predict(processed_image, model, device,topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    Arguments:
        processed_image: a processes image as returned by the 'process_image' function
        model: A Pytorch model
        device: On what device scoring should happen: CPU or GPU
        topk: Return the top k predictions
    '''
    import torch
    import numpy as np
    
    model.to(device)

    tensor = torch.from_numpy(np.expand_dims(processed_image, axis=0))
    tensor = tensor.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor.float())

    ps = torch.exp(outputs)
    probs, ind = ps.topk(topk)

    mapping = {v: k for k, v in model.class_to_idx.items()}

    classes = [mapping[i] for i in ind[0].tolist()]

    return probs[0].tolist(), classes
