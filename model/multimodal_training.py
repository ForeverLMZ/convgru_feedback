'''
Pared-down training script for testing basic functionality only
'''

import torch
import math
import pickle
import argparse


from ambiguous.dataset.dataset import DatasetFromNPY, DatasetTriplet

from scipy.special import softmax
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T

from utils.datagen import *
from model.newGraph import Graph, Architecture


parser = argparse.ArgumentParser()
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser.add_argument('--cuda', type = bool, default = True, help = 'use gpu or not')
parser.add_argument('--epochs', type = int, default = 50)
parser.add_argument('--layers', type = int, default = 1)
parser.add_argument('--topdown_c', type = int, default = 10)
parser.add_argument('--topdown_h', type = int, default = 10)
parser.add_argument('--topdown_w', type = int, default = 10)
parser.add_argument('--hidden_dim', type = int, default = 10)
parser.add_argument('--reps', type = int, default = 1)
parser.add_argument('--topdown', type = str2bool, default = True)
parser.add_argument('--connection_decay', type = str, default = 'ones')
parser.add_argument('--return_bottom_layer', type = str2bool, default = False)

parser.add_argument('--model_save', type = str, default = 'saved_models/data.pt')
parser.add_argument('--results_save', type = str, default = 'results/multi_input_data.npy')

args = vars(parser.parse_args())

# %%
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed for reproducibility
torch.manual_seed(42)
print(device)

# # 1: Prepare dataset
print('Loading datasets')

def test_sequence(dataloader):
    
    '''
    Inference
        :param dataloader
            dataloader to draw the target image from
        :param clean data (torchvision.Dataset)
            clean dataset to draw bottom-up sequence images from
        :param dataset_ref (list)
            if providing image clue, provide label reference as well
    '''
    correct = 0
    total = 0

    with torch.no_grad():

        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            x0 = x[:,0,:,:,:].float()
            x1 = x[:,1,:,:,:].float()
            x0 = torch.unsqueeze(x0, 1)
            x1 = torch.unsqueeze(x1, 1)
            optimizer.zero_grad()
            input_list = [x0,x1]
            output = model(input_list)

            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    #print('Accuracy of the network on the 10000 test images: %d %%' % (
    #    100 * correct / total))
    
    return correct/total

def train_sequence():
    running_loss = 0.0
        
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        x0 = x[:,0,:,:,:].float()
        x1 = x[:,1,:,:,:].float()
        x0 = torch.unsqueeze(x0, 1)
        x1 = torch.unsqueeze(x1, 1)
        optimizer.zero_grad()
        input_list = [x0,x1]
        output = model(input_list)

            
        loss = criterion(output, y)
        running_loss += loss.item()
            
        loss.backward()
        optimizer.step()
    
    return running_loss


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, mode):
        'Initialization'
        self.list_IDs = list_IDs
        self.mode = mode
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        root = '/home/mila/m/mingze.li/network/scratch/n/nizar.islah/amnistV5_seq_cache/'+ self.mode + '/' #TODO: move it outta here
        X = torch.load(root+ 'img_seq_' + str(ID) + '.pt')
        y = torch.load(root+'sum_label_'+ str(ID) + '.pt') #clear number
        return (X[:2], y[0])
    


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 1}
 

# Generators
training_set = Dataset(range(60168), mode = 'train')
train_loader = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(range(10862),mode = 'test')
test_loader = torch.utils.data.DataLoader(validation_set, **params)


connection_strengths = [1, 1, 1, 1] 
criterion = nn.CrossEntropyLoss()

#connections = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
#node_params = [(1, 28, 28, 5, 5), (10, 15, 15, 5, 5), (10, 9, 9, 3, 3), (10, 3, 3, 3, 3)]

input_nodes = [0,3] # V1
output_node = 3 #IT
input_dims = [1, 0, 0, 10]
input_sizes = [(28, 28), (0, 0), (0, 0), (10, 10)]
graph_loc = '/home/mila/m/mingze.li/code/convgru_feedback/nonlinear_graph.CSV'
graph = Graph(graph_loc, input_nodes, output_node=3)
#graph = Graph(connections = connections, 
#              conn_strength = connection_strengths, 
#              input_node_indices = input_node, 
#             output_node_index = output_node,
#              input_node_params = node_params,
#              dtype = torch.cuda.FloatTensor)
print("hello")
model = Architecture(graph, input_sizes, input_dims).cuda().float()

optimizer = optim.Adam(model.parameters(),lr = 0.00001)

losses = {'loss': [], 'train_acc': [], 'val_acc': []}    

if os.path.exists(args['model_save']):
    model.load_state_dict(torch.load(args['model_save']))
    print("Loading existing ConvGRU model")
else:
    print("No pretrained model found. Training new one.")

for epoch in range(args['epochs']):
    train_acc = test_sequence(train_loader)
    val_acc = test_sequence(test_loader)
    loss = train_sequence()
    
    printlog = '| epoch {:3d} | running loss {:5.4f} | train accuracy {:1.5f} |val accuracy {:1.5f}'.format(epoch, loss, train_acc, val_acc)

    print(printlog)
    losses['loss'].append(loss)
    losses['train_acc'].append(train_acc)
    losses['val_acc'].append(val_acc)

    with open(args['results_save'], "wb") as f:
        pickle.dump(losses, f)

    torch.save(model.state_dict(), args['model_save'])


