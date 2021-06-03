# %% [markdown]
# # Chest X-Ray Images (Pneumonia)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:00:40.722034Z","iopub.execute_input":"2021-06-03T04:00:40.722405Z","iopub.status.idle":"2021-06-03T04:00:42.251115Z","shell.execute_reply.started":"2021-06-03T04:00:40.722353Z","shell.execute_reply":"2021-06-03T04:00:42.250210Z"}}
import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F 
from torch import nn,optim
from torchvision import transforms as T,datasets,models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm
pd.options.plotting.backend = "plotly"

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:00:42.257805Z","iopub.execute_input":"2021-06-03T04:00:42.258151Z","iopub.status.idle":"2021-06-03T04:00:42.319722Z","shell.execute_reply.started":"2021-06-03T04:00:42.258111Z","shell.execute_reply":"2021-06-03T04:00:42.318620Z"}}
def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        # これを有効にしないと、計算した勾配が毎回異なり、再現性が担保できない。
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# デバイスを選択する。
device = get_device(use_gpu=True)

# %% [markdown]
# # Loading Dataset and Applying Transforms

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:00:42.321523Z","iopub.execute_input":"2021-06-03T04:00:42.322154Z","iopub.status.idle":"2021-06-03T04:00:42.329595Z","shell.execute_reply.started":"2021-06-03T04:00:42.322112Z","shell.execute_reply":"2021-06-03T04:00:42.328708Z"}}
data_dir = "../input/chest-xray-pneumonia/chest_xray/chest_xray"
TEST = 'test'
TRAIN = 'train'
VAL ='val'

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:00:42.331071Z","iopub.execute_input":"2021-06-03T04:00:42.331626Z","iopub.status.idle":"2021-06-03T04:00:42.339753Z","shell.execute_reply.started":"2021-06-03T04:00:42.331588Z","shell.execute_reply":"2021-06-03T04:00:42.339000Z"}}
def data_transforms(phase = None):
    
    if phase == TRAIN:

        data_T = T.Compose([
            
                T.Resize(size = (256,256)),
                T.RandomRotation(degrees = (-20,+20)),
                T.CenterCrop(size=224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    
    elif phase == TEST or phase == VAL:

        data_T = T.Compose([

                T.Resize(size = (224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
    return data_T

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:00:42.341200Z","iopub.execute_input":"2021-06-03T04:00:42.341799Z","iopub.status.idle":"2021-06-03T04:00:47.286674Z","shell.execute_reply.started":"2021-06-03T04:00:42.341746Z","shell.execute_reply":"2021-06-03T04:00:47.285856Z"}}
trainset = datasets.ImageFolder(os.path.join(data_dir, TRAIN),transform = data_transforms(TRAIN))
testset = datasets.ImageFolder(os.path.join(data_dir, TEST),transform = data_transforms(TEST))
validset = datasets.ImageFolder(os.path.join(data_dir, VAL),transform = data_transforms(VAL))
class_names = trainset.classes
print(class_names)
print(trainset.class_to_idx)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:00:47.288030Z","iopub.execute_input":"2021-06-03T04:00:47.288360Z","iopub.status.idle":"2021-06-03T04:00:47.295223Z","shell.execute_reply.started":"2021-06-03T04:00:47.288324Z","shell.execute_reply":"2021-06-03T04:00:47.294424Z"}}
def plot_class_count(classes,name = None):
    pd.DataFrame(classes,columns = [name]).groupby([classes]).size().plot(kind = 'bar',title = name).show()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-03T04:00:47.296796Z","iopub.execute_input":"2021-06-03T04:00:47.297051Z","iopub.status.idle":"2021-06-03T04:00:47.304967Z","shell.execute_reply.started":"2021-06-03T04:00:47.297020Z","shell.execute_reply":"2021-06-03T04:00:47.304074Z"},"jupyter":{"source_hidden":true}}
def get_class_count(dataset,name = None):
    classes = []
    for _,label in dataset:
        if label == 0:
            classes.append(class_names[label])
            
        elif label == 1:
            classes.append(class_names[label])
            
    return classes

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:00:47.309112Z","iopub.execute_input":"2021-06-03T04:00:47.309365Z","iopub.status.idle":"2021-06-03T04:02:54.027784Z","shell.execute_reply.started":"2021-06-03T04:00:47.309340Z","shell.execute_reply":"2021-06-03T04:02:54.026938Z"}}
trainset_class_count = get_class_count(trainset,name = 'trainset_classes_count')
plot_class_count(trainset_class_count,name = 'trainset_classes_count')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:02:54.029722Z","iopub.execute_input":"2021-06-03T04:02:54.030071Z","iopub.status.idle":"2021-06-03T04:03:08.148744Z","shell.execute_reply.started":"2021-06-03T04:02:54.030034Z","shell.execute_reply":"2021-06-03T04:03:08.147970Z"}}
testset_class_count = get_class_count(testset,name = 'testset_classes_count')
plot_class_count(testset_class_count,name = 'testset_classes_count')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:03:08.150205Z","iopub.execute_input":"2021-06-03T04:03:08.150535Z","iopub.status.idle":"2021-06-03T04:03:08.537255Z","shell.execute_reply.started":"2021-06-03T04:03:08.150499Z","shell.execute_reply":"2021-06-03T04:03:08.536471Z"}}
validset_class_count = get_class_count(validset,name = 'validset_classes_count')
plot_class_count(validset_class_count,name = 'validset_classes_count')

# %% [markdown]
# #Loading Dataset and Applying Transforms

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:03:08.538484Z","iopub.execute_input":"2021-06-03T04:03:08.538820Z","iopub.status.idle":"2021-06-03T04:03:08.543667Z","shell.execute_reply.started":"2021-06-03T04:03:08.538783Z","shell.execute_reply":"2021-06-03T04:03:08.542682Z"}}
trainloader = DataLoader(trainset,batch_size = 16,shuffle = True)
validloader = DataLoader(validset,batch_size = 8,shuffle = True)
testloader = DataLoader(testset,batch_size = 8,shuffle = True)

# %% [markdown]
# # Plot Image

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:03:08.545013Z","iopub.execute_input":"2021-06-03T04:03:08.545528Z","iopub.status.idle":"2021-06-03T04:03:08.554467Z","shell.execute_reply.started":"2021-06-03T04:03:08.545492Z","shell.execute_reply":"2021-06-03T04:03:08.553617Z"}}
def show_image(image,title = None,get_denormalize = False):
    
    image = image.permute(1,2,0)
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])
    
    image = image*std + mean
    image = np.clip(image,0,1)
    
    if get_denormalize == False:
        plt.figure(figsize=[15, 15])
        plt.imshow(image)

        if title != None:
            plt.title(title)
            
    else : 
        return image

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:03:08.555432Z","iopub.execute_input":"2021-06-03T04:03:08.555662Z","iopub.status.idle":"2021-06-03T04:03:09.396365Z","shell.execute_reply.started":"2021-06-03T04:03:08.555641Z","shell.execute_reply":"2021-06-03T04:03:09.395482Z"}}
dataiter = iter(trainloader)
images,labels = dataiter.next()

out = make_grid(images,nrow=4)

show_image(out, title=[class_names[x] for x in labels])

# %% [markdown]
# # Fine-Tuning VGG-16

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:03:09.397438Z","iopub.execute_input":"2021-06-03T04:03:09.397938Z","iopub.status.idle":"2021-06-03T04:03:11.699232Z","shell.execute_reply.started":"2021-06-03T04:03:09.397897Z","shell.execute_reply":"2021-06-03T04:03:11.698357Z"}}
model = models.vgg16()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:03:11.700748Z","iopub.execute_input":"2021-06-03T04:03:11.701104Z","iopub.status.idle":"2021-06-03T04:03:17.479134Z","shell.execute_reply.started":"2021-06-03T04:03:11.701069Z","shell.execute_reply":"2021-06-03T04:03:17.478361Z"}}

for param in model.parameters():
    param.requires_grad = False




classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096)),
                                         ('relu', nn.ReLU()),
                                         ('dropout',nn.Dropout(0.3)),
                                         ('fc2', nn.Linear(4096, 4096)),
                                         ('relu', nn.ReLU()),
                                         ('drop', nn.Dropout(0.3)),
                                         ('fc3', nn.Linear(4096, 2)), 
                                         ('output', nn.LogSoftmax(dim = 1))]))

model.classifier = classifier
model.to(device)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:03:17.480390Z","iopub.execute_input":"2021-06-03T04:03:17.480716Z","iopub.status.idle":"2021-06-03T04:03:17.485826Z","shell.execute_reply.started":"2021-06-03T04:03:17.480681Z","shell.execute_reply":"2021-06-03T04:03:17.485032Z"}}
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)
schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor = 0.1,patience = 5)
epochs = 15
valid_loss_min = np.Inf

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:03:17.487121Z","iopub.execute_input":"2021-06-03T04:03:17.487659Z","iopub.status.idle":"2021-06-03T04:33:28.599702Z","shell.execute_reply.started":"2021-06-03T04:03:17.487623Z","shell.execute_reply":"2021-06-03T04:33:28.598614Z"}}
record_train_loss2 = []
record_valid_loss2 = []
record_train_acc2 = []
record_valid_acc2 = []
def accuracy(y_pred,y_true):
    y_pred = torch.exp(y_pred)
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))
for i in range(epochs):
    
    train_loss = 0.0
    valid_loss = 0.0
    train_acc = 0.0
    valid_acc = 0.0 
    
    
    model.train()
    
    for images,labels in tqdm(trainloader):
        
        images = images.to(device)
        labels = labels.to(device)
        
        ps = model(images)
        loss = criterion(ps,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_acc += accuracy(ps,labels)
        train_loss += loss.item()
        
    avg_train_acc = train_acc / len(trainloader)
    avg_train_loss = train_loss / len(trainloader)
    record_train_loss2.append(avg_train_loss)
        
    model.eval()
    with torch.no_grad():
        
        for images,labels in tqdm(validloader):
            
            images = images.to(device)
            labels = labels.to(device)
            
            ps = model(images)
            loss = criterion(ps,labels)
            
            valid_acc += accuracy(ps,labels)
            valid_loss += loss.item()
            
            
        avg_valid_acc = valid_acc / len(validloader)
        avg_valid_loss = valid_loss / len(validloader)
        record_valid_loss2.append(avg_valid_loss)
        
        schedular.step(avg_valid_loss)
        
        if avg_valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).   Saving model ...'.format(valid_loss_min,avg_valid_loss))
            torch.save({
                'epoch' : i,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'valid_loss_min' : avg_valid_loss
            },'Pneumonia_model.pt')
            
            valid_loss_min = avg_valid_loss
            
            
    print("Epoch : {} Train Loss : {:.6f} Train Acc : {:.6f}".format(i+1,avg_train_loss,avg_train_acc))
    print("Epoch : {} Valid Loss : {:.6f} Valid Acc : {:.6f}".format(i+1,avg_valid_loss,avg_valid_acc))

# %% [markdown]
# # Training Model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:33:28.600983Z","iopub.execute_input":"2021-06-03T04:33:28.601430Z","iopub.status.idle":"2021-06-03T04:33:28.606700Z","shell.execute_reply.started":"2021-06-03T04:33:28.601392Z","shell.execute_reply":"2021-06-03T04:33:28.605794Z"}}
criterion = nn.NLLLoss() #criterion :正規化
optimizer = optim.Adam(model.parameters(),lr = 0.001)
schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor = 0.1,patience = 5)
epochs = 15
valid_loss_min = np.Inf

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:33:28.607927Z","iopub.execute_input":"2021-06-03T04:33:28.608446Z","iopub.status.idle":"2021-06-03T04:33:28.627183Z","shell.execute_reply.started":"2021-06-03T04:33:28.608408Z","shell.execute_reply":"2021-06-03T04:33:28.626345Z"}}
def accuracy(y_pred,y_true):
    y_pred = torch.exp(y_pred)
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:36:54.659419Z","iopub.execute_input":"2021-06-03T04:36:54.659840Z","iopub.status.idle":"2021-06-03T05:07:14.365160Z","shell.execute_reply.started":"2021-06-03T04:36:54.659800Z","shell.execute_reply":"2021-06-03T05:07:14.362261Z"}}
record_train_loss = []
record_train_acc = []
for i in range(epochs):
    
    train_loss = 0.0
    valid_loss = 0.0
    train_acc = 0.0
    valid_acc = 0.0 
    
    
    model.train()
    
    for images,labels in tqdm(trainloader):
        
        images = images.to(device)
        labels = labels.to(device)
        
        ps = model(images)
        loss = criterion(ps,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_acc += accuracy(ps,labels)
        train_loss += loss.item()
        
        
    avg_train_acc = train_acc / len(trainloader)
    avg_train_loss = train_loss / len(trainloader)
    record_train_acc.append(avg_train_acc)
    record_train_loss.append(avg_train_loss)
        
    model.eval()
    with torch.no_grad():
        
        for images,labels in tqdm(validloader):
            
            images = images.to(device)
            labels = labels.to(device)
            
            ps = model(images)
            loss = criterion(ps,labels)
            
            valid_acc += accuracy(ps,labels)
            valid_loss += loss.item()
            
            
        avg_valid_acc = valid_acc / len(validloader)
        avg_valid_loss = valid_loss / len(validloader)
        record_valid_loss.append(avg_valid_loss)
        
        schedular.step(avg_valid_loss)
        
        if avg_valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).   Saving model ...'.format(valid_loss_min,avg_valid_loss))
            torch.save({
                'epoch' : i,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'valid_loss_min' : avg_valid_loss
            },'Pneumonia_model.pt')
            
            valid_loss_min = avg_valid_loss
            
            
    print("Epoch : {} Train Loss : {:.6f} Train Acc : {:.6f}".format(i+1,avg_train_loss,avg_train_acc))
    print("Epoch : {} Valid Loss : {:.6f} Valid Acc : {:.6f}".format(i+1,avg_valid_loss,avg_valid_acc))

# %% [markdown]
# # Testset results

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T05:11:34.035512Z","iopub.execute_input":"2021-06-03T05:11:34.035894Z","iopub.status.idle":"2021-06-03T05:11:47.290814Z","shell.execute_reply.started":"2021-06-03T05:11:34.035848Z","shell.execute_reply":"2021-06-03T05:11:47.289231Z"}}
record_test_acc = []
record_test_loss = []
model.eval()

test_loss = 0
test_acc = 0
record_test_loss = []
for images,labels in testloader:
    
    images = images.to(device)
    labels = labels.to(device)
    
    pred = model(images)
    loss = criterion(pred,labels)
    
    test_loss += loss.item()
    test_acc += accuracy(pred,labels)
    record_test_acc.append(test_acc)
    record_test_loss.append(test_loss)
    
    
avg_test_loss = test_loss/len(testloader)
avg_test_acc = test_acc/len(testloader)


print("Test Loss : {:.6f} Test Acc : {:.6f}".format(avg_test_loss,avg_test_acc))

# %% [code] {"execution":{"iopub.status.busy":"2021-06-03T05:14:25.265841Z","iopub.execute_input":"2021-06-03T05:14:25.266174Z","iopub.status.idle":"2021-06-03T05:14:25.421460Z","shell.execute_reply.started":"2021-06-03T05:14:25.266141Z","shell.execute_reply":"2021-06-03T05:14:25.420620Z"},"jupyter":{"outputs_hidden":false}}



plt.plot(range(len(record_train_loss)), record_train_loss, label="Train")


plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-03T05:14:58.270148Z","iopub.execute_input":"2021-06-03T05:14:58.270468Z","iopub.status.idle":"2021-06-03T05:14:58.424518Z","shell.execute_reply.started":"2021-06-03T05:14:58.270438Z","shell.execute_reply":"2021-06-03T05:14:58.423681Z"},"jupyter":{"outputs_hidden":false}}
L = len(record_train_acc)
plt.plot(range(L), record_train_acc, label="Train")


plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-03T05:28:13.985199Z","iopub.execute_input":"2021-06-03T05:28:13.985523Z","iopub.status.idle":"2021-06-03T05:28:13.990736Z","shell.execute_reply.started":"2021-06-03T05:28:13.985493Z","shell.execute_reply":"2021-06-03T05:28:13.989460Z"},"jupyter":{"outputs_hidden":false}}

print("Test Acc : {:.6f}".format(avg_test_acc))

# %% [markdown]
# 

# %% [markdown]
# # Predict Plotting

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T05:42:35.560483Z","iopub.execute_input":"2021-06-03T05:42:35.560814Z","iopub.status.idle":"2021-06-03T05:42:35.567273Z","shell.execute_reply.started":"2021-06-03T05:42:35.560777Z","shell.execute_reply":"2021-06-03T05:42:35.566329Z"}}
def view_classify(img,ps,label):
    
    class_name = ['NORMAL', 'PNEUMONIA']
    classes = np.array(class_name)

    ps = ps.cpu().data.numpy().squeeze()
    img = show_image(img,get_denormalize = True)
    
    

    fig, (ax1, ax2) = plt.subplots(figsize=(8,12), ncols=2)
    ax1.imshow(img)
    ax1.set_title('Ground Truth : {}'.format(class_name[label]))
    ax1.axis('off')
    ax2.barh(classes, ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Predicted Class')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    return None

# %% [markdown]
# # Example

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T05:42:39.745917Z","iopub.execute_input":"2021-06-03T05:42:39.746226Z","iopub.status.idle":"2021-06-03T05:42:40.016959Z","shell.execute_reply.started":"2021-06-03T05:42:39.746200Z","shell.execute_reply":"2021-06-03T05:42:40.016253Z"}}
image,label = testset[0]

ps = torch.exp(model(image.to(device).unsqueeze(0))) 
view_classify(image,ps,label)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:34:25.812605Z","iopub.status.idle":"2021-06-03T04:34:25.813207Z"}}
image,label = testset[36]

ps = torch.exp(model(image.to(device).unsqueeze(0)))
view_classify(image,ps,label)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:34:25.814353Z","iopub.status.idle":"2021-06-03T04:34:25.814927Z"}}
image,label = testset[56]

ps = torch.exp(model(image.to(device).unsqueeze(0)))
view_classify(image,ps,label)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-03T04:34:25.816114Z","iopub.status.idle":"2021-06-03T04:34:25.816688Z"}}
image,label = testset[110]

ps = torch.exp(model(image.to(device).unsqueeze(0)))
view_classify(image,ps,label)
            

