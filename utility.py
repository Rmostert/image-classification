def load_data(data_dir):
    
    '''
    Load and transforms the data to be used in the Pytorch model
    Arguments:
        data_dir: Directory the data reside in. Needs to have the following structure
            train
            valid
            test
    '''

    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torchvision.models as models
    from torch import optim
    from torch import nn
    from collections import OrderedDict

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # Define your transforms for the training, validation, and testing sets

    means = [0.485, 0.456, 0.406]
    stdevs = [0.229, 0.224, 0.225]


    train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=means, std=stdevs)])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=means, std=stdevs)])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=means, std=stdevs)])

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(root=train_dir,transform=train_transforms)
    valid_datasets = datasets.ImageFolder(root=valid_dir,transform=validation_transforms)
    test_datasets = datasets.ImageFolder(root=test_dir,transform=test_transforms)

    # Using the image datasets and the tranforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(dataset=train_datasets,batch_size=32,shuffle=True)
    validloader = torch.utils.data.DataLoader(dataset=valid_datasets,batch_size=32)
    testloader = torch.utils.data.DataLoader(dataset=test_datasets,batch_size=32)

    class_to_idx = train_datasets.class_to_idx

    return trainloader,validloader,testloader,class_to_idx


def load_checkpoint(filepath):
    
    '''
    Loads the model checkpoint file to be used in prediction
    Arguments:
        filepath: file path where the checkpoint reside
    '''
    
    import torch
    import torchvision.models as models
    from torch import nn
    from collections import OrderedDict
    

    checkpoint = torch.load(filepath,map_location='cpu')

    architecture = checkpoint['architecture']
    hidden_units = checkpoint['hidden_units']
    output_size = checkpoint['output_size']

    if architecture == 'vgg16':
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(0.5)),
                              ('fc2', nn.Linear(hidden_units, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
        model = models.vgg16(pretrained=True)

    elif architecture == 'densenet121':
        classifier = nn.Sequential(OrderedDict([
                                      ('fc1', nn.Linear(1024, hidden_units)),
                                      ('relu', nn.ReLU()),
                                      ('dropout1', nn.Dropout(0.5)),
                                      ('fc2', nn.Linear(hidden_units, output_size)),
                                      ('output', nn.LogSoftmax(dim=1))
                                      ]))
        model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    import numpy as np

    # Resize
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))

    w, h = image.size

    left = (w - 224)/2
    top = (h - 224)/2
    right = (w + 224)/2
    bottom = (h + 224)/2

    cropped_img = image.crop((left, top, right, bottom))
    np_image = np.array(cropped_img)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm_img = (np_image - mean) / std

    final_image = norm_img.transpose((2, 0, 1))

    return final_image
