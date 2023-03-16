from model.topdown_gru import ConvGRUTopDownCell
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import math

#def conv_output_size(input_size, kernel_size, stride=1):
#    padding = kernel_size[0] // 2
#    return ((input_size - kernel_size) // stride) + 1

class Node:
    def __init__(self, index, input_nodes, output_nodes, input_size = (28, 28), input_dim = 1, hidden_dim = 10, kernel_size = (3,3)):
        self.index = index
        self.in_nodes_indices = input_nodes #nodes passing values into current node #contains Node index (int)
        self.out_nodes_indices = output_nodes #nodes being passed with values from current node #contains Node index (int)
        self.in_strength = [] #connection strengths of in_nodes
        self.out_strength = [] #connection strength of out_nodes
        self.rank_list = [-42] #default value. if the Node end up with only -42 as its rank_list element, then 


        #cell params
        self.input_size = input_size #Height and width of input tensor as (height, width).
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size 

    def __eq__(self, other): 
        return self.index == other.index


class Graph(object):
    """ 
    A brain architecture graph object, directed by default. 
    
    :param connections (n x n)
        directed binary matrix of connections row --> column 
    :param connection_strength
    """
    def __init__(self, graph_loc, input_nodes, output_node, directed=True, bias=False, dtype = torch.cuda.FloatTensor):
        
        graph_df = pd.read_csv(graph_loc)
        self.num_node = graph_df.shape[0]
        self.conn_strength = torch.tensor(graph_df.iloc[:, :self.num_node].values)   # connection_strength_matrix
        self.conn = self.conn_strength > 0                                           # binary matrix of connections
        self.node_dims = [row for _, row in graph_df.iterrows()]                     # dimensions of each node/area
        self.nodes = self.generate_node_list(self.conn, self.node_dims)              # turn dataframe into list of graph nodes
        
        self.input_node_indices = input_nodes
        self.output_node_index = output_node

        self.directed = directed #flag for whether the connections are directed 

        self.dtype = dtype
        self.bias = bias

        for input_node in self.input_node_indices:
            self.rank_node(self.nodes[input_node],self.nodes[self.output_node_index], 0, 0)

        #for node in range (self.num_node):
            
            #print(self.nodes[node].rank_list)

    #def calculate_parameters(self, input_size, output_size): #why did my code disapear ughh now I have to rewrite again
        #follows the following formula:: output_size = floor((input_height + padding - kernal) / stride) + 1
         
            

    def generate_node_list(self, connections, node_dims):
        nodes = []
        # initialize node list
        for n in range(self.num_node):
            input_nodes = torch.nonzero(connections[:, n])
            output_nodes = torch.nonzero(connections[n, :])
            hidden_dim = node_dims[n].hidden_dim
            input_dim = node_dims[n].input_dim
            input_size = (node_dims[n].input_h, node_dims[n].input_w)
            kernal_size = (node_dims[n].kernal_h, node_dims[n].kernal_w)
            
            node = Node(n, input_nodes, output_nodes, input_dim=input_dim, input_size=input_size, hidden_dim=hidden_dim)
            nodes.append(node)
            
        return nodes
    
    def rank_node(self, current_node, output_node_index, rank_val, num_pass):
        ''' ranks each node in the graph'''
        current_node.rank_list.append(rank_val)
        rank_val = rank_val + 1
        for node in current_node.out_nodes_indices:
            num_pass = num_pass + 1
            if (current_node == output_node_index and num_pass == self.num_edge):
                self.max_rank = current_node.rank.max()
                return
            else:
                self.rank_node(self.nodes[node], output_node_index, rank_val, num_pass)

    def build_architecture(self):
        architecture = Architecture(self).cuda().float()
        return architecture

    #def find_feedforward_cells(self, node):
        #return self.nodes[node].in_nodes_indices
    def find_feedforward_cells(self, node, t):
        in_nodes = []
        for n in self.nodes[node].in_nodes_indices:
            if ((t-1) in self.nodes[n].rank_list):
                in_nodes.append(n)
        return in_nodes

    def find_feedback_cells(self, node, t):
        return self.nodes[node].out_nodes_indices
        # if ((t+1) in self.nodes[n].rank_list):
        #     in_nodes.append(n)
        # return in_nodes

    def find_longest_path_length(self): #TODO: 
        return 42
        #I think this is not needed but we'll see

class Architecture(nn.Module):
    def __init__(self, graph, input_sizes, img_channel_dims, output_size=10, dropout=True, dropout_p=0.25, rep=1):
        '''
        :param graph (Graph object)
            object containing architecture graph
        :param input_sizes (list of tuples)
            list of input sizes (height, width) per node. If a node doesn't receive an outside input, it's (0, 0)
        :param img_channel_dims (list of ints)
            list of input dimension sizes per node. If a node doesn't receive an outside input, it's 0
        :param dropout (bool)
        :param rep (int)
            time steps to revisit the sequence
        '''
        super(Architecture, self).__init__()

        self.graph = graph
        self.rep = rep # repetition, i.e how many times to view the sequence in full
        self.use_dropout = dropout
        self.dropout = nn.Dropout(dropout_p)

        cell_list = []
        for node in range(graph.num_node):
            cell_list.append(ConvGRUTopDownCell(input_size=((graph.nodes[node].input_size[0], graph.nodes[node].input_size[1])), 
                                         input_dim=graph.nodes[node].input_dim,
                                         topdown_dim= 1, #MNIST dataset
                                         hidden_dim = int(graph.nodes[node].hidden_dim),
                                         kernel_size=graph.nodes[node].kernel_size,
                                         bias=graph.bias,
                                         dtype=graph.dtype))
        self.cell_list = nn.ModuleList(cell_list)
        
        # Dimensionality of output layer
        h, w = self.graph.nodes[self.graph.output_node_index].input_size
        self.fc1 = nn.Linear(h * w * self.graph.nodes[self.graph.output_node_index].hidden_dim, 100) 
        self.fc2 = nn.Linear(100, output_size)
        

        # PROJECTIONS FOR INPUT
        #assuming all cells have the same input dim and the same hidden dim
        #input_conv_list = []
        #for input in range(len(self.graph.input_node_indices)):
        #    input_conv_list.append(nn.Conv2d(in_channels= self.graph.nodes[self.graph.input_node_indices[input]].input_dim,
        #                        out_channels= self.graph.nodes[self.graph.input_node_indices[input]].hidden_dim,
        #                        kernel_size=3,
        #                        padding=1,
        #                        device = 'cuda'))
        #self.input_conv_list = nn.ModuleList(input_conv_list)
        
        # PROJECTIONS FOR BOTTOM-UP INTERLAYER CONNECTIONS
        bottomup_projections = []
        
        for node in range(graph.num_node):
            # dimensions of ALL bottom-up inputs to current node
            #all_input_h = sum([conv_output_size(graph.nodes[i.item()].input_size[0], graph.nodes[i.item()].kernel_size[0]) 
            #                   for i in self.graph.nodes[node].in_nodes_indices]) + input_sizes[node][0]
            #all_input_w = sum([conv_output_size(graph.nodes[i.item()].input_size[1], graph.nodes[i.item()].kernel_size[1]) 
            #                  for i in self.graph.nodes[node].in_nodes_indices]) + input_sizes[node][1]
            
            all_input_h = sum([graph.nodes[i.item()].input_size[0] for i in self.graph.nodes[node].in_nodes_indices]) + input_sizes[node][0]
            all_input_w = sum([graph.nodes[i.item()].input_size[1] for i in self.graph.nodes[node].in_nodes_indices]) + input_sizes[node][1]
            all_input_dim = sum([graph.nodes[i.item()].hidden_dim for i in self.graph.nodes[node].in_nodes_indices]) + img_channel_dims[node]
            target_h, target_w = graph.nodes[node].input_size
            
            proj = nn.Linear(all_input_dim*all_input_h*all_input_w, graph.nodes[node].input_dim*target_h*target_w)
            bottomup_projections.append(proj)
            
        self.bottomup_projections = nn.ModuleList(bottomup_projections)
        
        
        # PROJECTIONS FOR TOPDOWN INTERLAYER CONNECTIONS
        topdown_gru_proj = []
        for node in range(self.graph.num_node):
            # dimensions of ALL top-down inputs to current node
            #all_input_h = sum([conv_output_size(graph.nodes[i.item()].input_size[0], graph.nodes[i.item()].kernel_size[0]) 
            #                   for i in self.graph.nodes[node].out_nodes_indices]) + input_sizes[node][0]
            #all_input_w = sum([conv_output_size(graph.nodes[i.item()].input_size[1], graph.nodes[i.item()].kernel_size[1]) 
            #                   for i in self.graph.nodes[node].out_nodes_indices]) + input_sizes[node][1]
            
            all_input_h = sum([graph.nodes[i.item()].input_size[0] for i in self.graph.nodes[node].out_nodes_indices])
            all_input_w = sum([graph.nodes[i.item()].input_size[1] for i in self.graph.nodes[node].out_nodes_indices])
            all_input_dim = sum([graph.nodes[i.item()].input_dim for i in self.graph.nodes[node].out_nodes_indices])
            
            # dimensions accepted by current node
            target_h, target_w = graph.nodes[node].input_size
            target_c = graph.nodes[node].hidden_dim + graph.nodes[node].input_dim
            
            proj = nn.Linear(all_input_dim*all_input_h*all_input_w, target_c*target_h*target_w)
            topdown_gru_proj.append(proj)
        self.topdown_projections = nn.ModuleList(topdown_gru_proj)

    def forward(self, all_inputs):
        """
        :param a list of tensor of size n, each consisting a input_tensor: (b, t, c, h, w). 
            n as the number of input signals. Order should correspond to self.graph.input_node_indices
        :param topdown: size (b,hidden,h,w) 
        :return: label 
        """
        #find the time length of the longest input signal + enough extra time for the last input to go through all nodes
        seq_len = max([i.shape[1] for i in all_inputs])
        process_time = seq_len + self.graph.num_node - 1
        batch_size = all_inputs[0].shape[0]
        hidden_states = self._init_hidden(batch_size=batch_size)
        hidden_states_prev = self._init_hidden(batch_size=batch_size)
        #time = seq_len * self.rep
        
        # Each time you look at the sequence
        for rep in range(self.rep):
            
            # For each image
            for t in range(process_time):
                print("time: ",t)

                # Go through each node and see if anything should be processed
                for node in range(self.graph.num_node):
                    print('node:',node,":")

                    # input size is same for all bottomup and topdown input for ease of combining inputs
                    c = self.graph.nodes[node].input_dim
                    h, w = self.graph.nodes[node].input_size
                    current_input_cells = self.graph.find_feedforward_cells(node, t) #computes cells that send information to current cell at the current timestep
                    ########################################
                    # Find bottomup inputs
                    bottomup = []
                    bottomup_info = False

                    # direct stimuli if (node receives it + the sequence isn't done)
                    if node in self.graph.input_node_indices and (t < seq_len):
                        stimuli = all_inputs[self.graph.input_node_indices.index(node)]
                        #bottomup.append(stimuli[:, t, :, :, :].flatten(start_dim=1))
                        bottomup.append(stimuli[:, t, :, :, :])
                        bottomup_info = True
                        only_stimuli = True
                        #bottomup.append(inp[:, t, :, :, :])

                    # bottom-up input from other nodes
                    for i, bottomup_node in enumerate(current_input_cells):
                        bottomup.append(hidden_states_prev[bottomup_node].flatten(start_dim=1))
                        #bottomup.append(hidden_states_prev[bottomup_node])
                        bottomup_info = True
                        only_stimuli = False
                        #bottomup.append(hidden_states_prev[bottomup_node]
                        #bottomup.append(hidden_states[bottomup_node])
                    
                    if bottomup_info == False:
                        continue
                        
                    #I don't know better ways to count number of elements in in_nodes_indices
                    num_in_nodes = 0
                    for i in self.graph.nodes[node].in_nodes_indices:
                        num_in_nodes += 1
                    
                    
                    if(len(current_input_cells) < num_in_nodes):
                        dummy_dim = [0,0,0]
                        
                        for in_nodes in self.graph.nodes[node].in_nodes_indices:
                            only_stimuli = False

                            dummy_dim[0] += self.graph.nodes[in_nodes.item()].hidden_dim
                            dummy_dim[1] += self.graph.nodes[in_nodes.item()].input_size[0]
                            dummy_dim[2] += self.graph.nodes[in_nodes.item()].input_size[1]
                        bottomup = torch.cat(bottomup, dim=1)
                        if (not only_stimuli):
                            bottomup = torch.cat((bottomup, torch.zeros((batch_size,(dummy_dim[0]*dummy_dim[1]*dummy_dim[2] - bottomup.shape[1])), device=torch.device('cuda'))), dim=1) #pads zeros
                    else:
                        bottomup = torch.cat(bottomup, dim=1).flatten(start_dim = 1)
                        
                        if (only_stimuli == False):
                        #calculates the expected size and pads zeros
                            dummy_dim = [0,0,0]
                            for in_nodes in self.graph.nodes[node].in_nodes_indices:
                                dummy_dim[0] += self.graph.nodes[in_nodes.item()].hidden_dim
                                dummy_dim[1] += self.graph.nodes[in_nodes.item()].input_size[0]
                                dummy_dim[2] += self.graph.nodes[in_nodes.item()].input_size[1]
                            print(dummy_dim[0],dummy_dim[1],dummy_dim[2])
                            bottomup = torch.cat((bottomup, torch.zeros((batch_size,(dummy_dim[0]*dummy_dim[1]*dummy_dim[2] - bottomup.shape[1])), device=torch.device('cuda'))), dim=1) #pads zeros
                        
                        
                                                # if there's no new info, skip rest of loop
                    #if not bottomup or torch.count_nonzero(torch.cat(bottomup, dim=1)) == 0:
                   # if bottomup.shape[1] == 0:
                        #continue
                    
                    # else concatenate all inputs and project to correct size
                    bottomup = self.bottomup_projections[node](bottomup).reshape(batch_size, c, h, w)
                    
                    ##################################
                    # Find topdown inputs, assumes every feedforward connection out of node has feedback
                    # Note there's no external topdown input
                    topdown = []

                    for i, topdown_node in enumerate(self.graph.nodes[node].out_nodes_indices):
                        topdown.append(hidden_states_prev[topdown_node].flatten(start_dim=1))
                        #print(topdown[0].shape)
                    
                    if not topdown or torch.count_nonzero(torch.cat(topdown, dim=1)) == 0:
                        #topdown = None
                        topdown = torch.zeros((hidden_states_prev[node].shape[0], self.graph.nodes[node].input_dim + hidden_states_prev[node].shape[1], self.graph.nodes[node].input_size[0], self.graph.nodes[node].input_size[1])).to('cuda')
                    else:
                        topdown = self.topdown_projections[node](torch.cat(topdown, dim=1))
                        topdown = topdown.reshape(batch_size, self.graph.nodes[node].input_dim + self.graph.nodes[node].hidden_dim, h, w)
                    
                    #################################################
                    # Finally, pass to layer
                    
                    #print('bottomup',bottomup.shape)
                    #print('hidden_states',hidden_states[node].shape)
                    #print('topdown',topdown.shape)
                    h = self.cell_list[node](bottomup, hidden_states[node], topdown)
                    if self.use_dropout:
                        h = self.dropout(h)
                    hidden_states[node] = h
                hidden_states_prev = hidden_states

        pred = self.fc1(F.relu(torch.flatten(hidden_states[self.graph.output_node_index], start_dim=1)))
        pred = self.fc2(F.relu(pred))
          
        return pred

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.graph.num_node):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states
