#import GNN_playground #this is only locally important
#print('GNN_playground durchgelaufen')
import BaShapes_Model as bsm
from BaShapes_Hetero import create_hetero_ba_houses
from graph_generation import create_graphs_for_heterodata, add_features_and_predict_outcome, compute_accu, compute_prediction_ce, compute_confusion_for_ce_line
from ce_generation import create_graphdict_from_ce, length_ce, create_test_ce, generate_cedict_from_ce
from ce_generation import create_random_ce_from_BAHetero, create_random_ce_from_metagraph, remove_front
import torch
import statistics
# from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
import torch as th
import os.path as osp
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG, DBLP
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, to_hetero
import torch_geometric
from torch_geometric.data import HeteroData
from random import randint
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import dgl
import colorsys
import random
import os
from owlapy.model import OWLObjectProperty, OWLObjectSomeValuesFrom
from owlapy.model import OWLDataProperty
from owlapy.model import OWLClass, OWLClassExpression
from owlapy.model import OWLDeclarationAxiom, OWLDatatype, OWLDataSomeValuesFrom, OWLObjectIntersectionOf, OWLEquivalentClassesAxiom, OWLObjectUnionOf
from owlapy.model import OWLDataPropertyDomainAxiom
from owlapy.model import IRI
from owlapy.render import DLSyntaxObjectRenderer
import warnings
import sys
import copy
import pandas as pd

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*findfont.*")
dlsr = DLSyntaxObjectRenderer()
random_seed = 3006
random.seed(random_seed)




# Paper: Heterogeneous Attention Network
path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
# We initialize conference node features with a single one-vector as feature:
target_category_DBLP = 'conference'
dataset = DBLP(path, transform=T.Constant(node_types=target_category_DBLP))
target_category_DBLP = 'author'  # we want to predict classes of author
# 4 different classes for author:
#   database, data mining, machine learning, information retrieval.
data = dataset[0]
#print('Dataset we are working with:', data)


# #now: Set up a hetero-graphsage
# for this: setup class for optimization
# Code: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hetero_conv_dblp.py 
 
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = {key: F.leaky_relu(x) 
            for key, x in conv(x_dict, edge_index_dict).items()}
        return self.lin(x_dict['author'])


# TODO: out_channels = number of classes of dataset
model = HeteroGNN(data.metadata(), hidden_channels=32, out_channels=4,
                  num_layers=3)
#model = to_hetero(model, data.metadata(), aggr='sum')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datadblp, modeldblp = data.to(device), model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.001)


# TODO: Rename in train_epoch
def train_epoch():
    model.train()
    optimizer.zero_grad()
    out = modeldblp(datadblp.x_dict, datadblp.edge_index_dict)
    mask = data['author'].train_mask
    loss = F.cross_entropy(out[mask], datadblp['author'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = modeldblp(datadblp.x_dict, datadblp.edge_index_dict).argmax(dim=-1)
    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = datadblp['author'][split]
        acc = (pred[mask] == datadblp['author'].y[mask]).sum() / mask.sum()
        #acc = (pred[mask] == data['author'].y[mask]).sum() / mask.size(dim=-1)
        accs.append(float(acc))
    return accs


# TODO: put this in some function
# TODO: Rename into train_model
def train_model():
    print('started training for ', modeldblp)
    modeldblp.train()
    for epoch in range(1, 100):
        loss = train_epoch()
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
              

# TODO: put training into function
# save model
retrain = False
path_name_saved = "content/models/"+'DBLP'
is_file_there = osp.isfile(path_name_saved) 
if(is_file_there == True and retrain == False):
    print("using saved model")
    modeldblp.load_state_dict(torch.load(path_name_saved))
else:
    print('training new model')
    train_model()
    print('new model is trained')
    PATH = "content/models/" + 'DBLP'
    print("File will be saved to: ", PATH)
    torch.save(model.state_dict(), PATH)
# evaluate accuracy
modeldblp.eval()  # evaluation-modus of model
#UNDO acc = test()[2]
#UNDO print(acc)
# OUTPUT of HGNN on DBLP: which research-area is the author in?
# TODO: put into functions
target = 'author'


#utils # From https://stackoverflow.com/questions/13852700
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def delete_files_in_folder(folder_path):
    """Deletes all files in the specified folder."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")


# visualize best num_top results from saved list, do not comment for usage
# TODO: output num_top results additionally
def graphdict_and_features_to_heterodata(graph_dict, features_list = []):
    hdata = HeteroData()
    # create features and nodes
    for name_tuple in features_list:
        name = name_tuple[0]
        hdata[name].x = name_tuple[1]
    # create edges
    # read from dict
    for edge in graph_dict:
        hdata[edge[0], edge[1], edge[2]].edge_index = torch.tensor([graph_dict[edge][0].tolist(), 
                                            graph_dict[edge][1].tolist()], dtype=torch.long)
    return hdata


def select_ones(numbers, num_ones_to_keep):
    ones_indices = [i for i, num in enumerate(numbers) if num == 1]
    
    if len(ones_indices) <= num_ones_to_keep:
        return numbers
    
    zeros_indices = random.sample(ones_indices, len(ones_indices) - num_ones_to_keep)
    
    for index in zeros_indices:
        numbers[index] = 0
    
    return numbers


def ce_fidelity(ce_for_fid, modelfid, datasetfid, node_type_expl, label_expl = -1): 
    fid_result = -1
    mask = datasetfid[node_type_expl]['test_mask']
    mask_tf = 0
    for value in mask.tolist():
        if str(value) == 'True' or str(value) == 'False':
            mask_tf = 1
            break
    metagraph = datasetfid.to_dict()
    ### falls node_type_expl == -1: Ändere dies auf das letzte aller möglichen labels
    if label_expl == -1:
        print(204, datasetfid[node_type_expl].y)
        list_labels = datasetfid[node_type_expl].y
        label_expl = max(set(list_labels))
    modelfid.eval()
    pred = modelfid(datasetfid.x_dict, datasetfid.edge_index_dict).argmax(dim=-1)
    pred_list = pred.tolist()
    for index, value in enumerate(pred_list):
        if value != label_expl:
            pred_list[index] = 0
        else:
            pred_list[index] = 1
    print(213, pd.Series(pred_list).value_counts())
    pred = torch.tensor(pred_list)
    if mask_tf == 0:
        mask = datasetfid[node_type_expl]['test_mask']
        print(222, len(mask.tolist())) #tensor of indices
        #print(205, label_expl, pred, len(pred.tolist()), pred.tolist().count(1), mask.tolist().count(True), mask)
        cedict = generate_cedict_from_ce(ce_for_fid)
        #mask = select_ones(mask, 100)
        # create new vector with samples only as true vector
        random.seed(236)
        smaller_mask = random.sample(mask.tolist(), k=min(200, len(mask.tolist())))
        mask = torch.tensor(smaller_mask)
    else:
        print('252, dblp')
        indices_of_ones = [i for i, value in enumerate(mask.tolist()) if value == True]
        random.seed(257)
        chosen_indices = random.sample(indices_of_ones, k=min(200, len(indices_of_ones)))
        mask = [i if i in chosen_indices else 0 for i in range(len(mask.tolist()))]
        mask = [x for x in mask if x != 0]
        mask = torch.tensor(mask)
    count_fids = 0
    count_zeros_test = 0
    count_zeros_gnn = 0
    for index in mask.tolist():
        print(232, index, pred[0])
        cedict = generate_cedict_from_ce(ce_for_fid)
        result_ce_fid = compute_prediction_ce(cedict, metagraph, node_type_expl, index)
        if pred[index] ==result_ce_fid:
            count_fids +=1
        if result_ce_fid == 0:
            count_zeros_test +=1
        if pred[index] == 0:
            count_zeros_gnn +=1
    print(226, 'zeros counted CE, GNN: ', count_zeros_test, count_zeros_gnn)
    fid_result = round(float(count_fids) / float(len(mask.tolist())),2)
    return fid_result


def available_edges_with_nodeid(graph, current_type, current_id, edgetype = 'to'):
    # graph is in dictionary form
    list_result = list()
    for key, value in graph.items():
        if key[0] == current_type and key[1] == edgetype:
            for _, indexvalue in enumerate(value[0].tolist()):
                if current_id == indexvalue:
                    list_result.append([key[2], value[1].tolist()[_], value[2][_]])
    return list_result
    

# TODO: Think of cases, where this could not work: How would we find the edge 1-1 in the house, if there are no 2-3 edges ?
def ce_confusion_iterative(ce, graph, current_graph_node): # current_graph_node is of form ['3',0], ['2',0]. ['2',1], etc. [nodetype, nodeid_of_nodetype]
    # TODO: Insert 'abstract current nodes', if a edge to a node not in the graph or without specified nodetype is called
    # save the number of abstract edges used (later, maybe instead of True / False as feedback ?
    
    
    result = set()
    if isinstance(ce, OWLClass):
        print(282, 'is class')
        if current_graph_node[0] != remove_front(ce.to_string_id()):
            print(283, 'class false', current_graph_node)
            return result, False, current_graph_node
        else:
            print(286, current_graph_node)
            return result, True, current_graph_node
    elif isinstance(ce, OWLObjectProperty):
        edgdetype = remove_front(ce.to_string_id())
        available_edges = available_edges_with_nodeid(graph, current_graph_node[0], current_graph_node[1], edgetype) #form should be [edge, endnodetype, endnodeid]
        if len(available_edges) > 0:
                # TODO: Add this edge to result, as the edge has been found
            #result.update()
            # retrieve all available edges
            set_possible_edges = set()
            for aved in available_edges:
                set_possible_edges.update(aved[2]) 
            for edgeind in set_possible_edges:
                if edgeind not in result:
                    result.update(edgeind)
                    break
            return result, True, current_graph_node
        return result, False, current_graph_node
    elif isinstance(ce, OWLObjectSomeValuesFrom):
        print(305, 'is edge with prop')
        edgetype = remove_front(ce._property.to_string_id())
        available_edges = available_edges_with_nodeid(graph, current_graph_node[0], current_graph_node[1], edgetype) #form should be [edge, endnodetype, endnodeid]
        current_best_length = len(result)
        result_copy = copy.deepcopy(result)
        local_result = set()
        local_current_grnd = current_graph_node
        some_edgewas_true = False
        for aved in available_edges:
            local_result = set()
            for i in result_copy:
                local_result.update(set(i))
            local_result.add(aved[2])
            feed1, feed2, current_graph_node = ce_confusion_iterative(ce._filler, graph, [aved[0], aved[1]])
            print(312, feed1, feed2, ce._filler)
            '''
            if feed2 and len(feed1)>=current_best_length:
                print(319, feed1)
                current_best_length = len(feed1)
                local_result_intern = feed1
                local_current_grnd = current_graph_node
            '''
            if feed2:
                some_edgewas_true = True
                print(319, feed1, current_graph_node)
                current_best_length = len(feed1)
                local_result_intern = feed1
                local_current_grnd = current_graph_node
                return set(list(local_result)+list(local_result_intern)), True, local_current_grnd
        if some_edgewas_true == False:
            print(340, current_graph_node, result)
            current_graph_node = 'abstract'
            return result, True, current_graph_node
            
            
        #if current_best_length >0:
        #    return set(list(local_result)+list(local_result_intern)), True, local_current_grnd
        #else:
        #    return result, False, current_graph_node
    elif isinstance(ce, OWLObjectIntersectionOf):
        return_truth = True
        for op in ce.operands(): #TODO: First select class if available, then further edges
            if isinstance(op, OWLClass) == True: 
                feed1, feed2, current_graph_node = ce_confusion_iterative(op, graph, current_graph_node)
                print(331, feed1)
                if feed1 != None:
                    result.update(list(feed1))
                if feed2 == False:
                    return_truth = False
        for op in ce.operands():
            if isinstance(op, OWLClass) == False:
                feed1, feed2, current_graph_node = ce_confusion_iterative(op, graph, current_graph_node)
                if feed1 != None:
                    result.update(list(feed1))
                if feed2 == False:
                    return_truth = False
        return result, return_truth, current_graph_node
    else:
        print(300, 'Some case is not implemented')
        print(301, ce)
        return result, False, current_graph_node
    return result, False, current_graph_node




def ce_confusion(ce,  motif = 'house'):
    motifgraph = dict()
    if motif == 'house':
        motifgraph = {('3', 'to', '2') :(torch.tensor([0,0], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long), [0,1]),
                     ('2', 'to', '3') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,0], dtype = torch.long), [1,0]),
                      ('2', 'to', '1') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long), [2,3]),
                     ('1', 'to', '2') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long), [3,2]),
                      ('2', 'to', '2') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([1,0], dtype = torch.long),[4,4]),
                       ('1', 'to', '1') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([1,0], dtype = torch.long),[5,5]),
                      # now the abstract class is included
                      ('0', 'to', 'abstract') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long), [-1]),
                     ('abstract', 'to', '0') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long), [-1]),
                      ('1', 'to', 'abstract') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,0], dtype = torch.long), [-1,-1]),
                     ('abstract', 'to', '1') :(torch.tensor([0,0], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long), [-1,-1]),
                      ('2', 'to', 'abstract') :(torch.tensor([0,1], dtype = torch.long), 
            torch.tensor([0,0], dtype = torch.long), [-1,-1]),
                     ('abstract', 'to', '2') :(torch.tensor([0,0], dtype = torch.long), 
            torch.tensor([0,1], dtype = torch.long), [-1,-1]),
                      ('3', 'to', 'abstract') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long), [-1]),
                     ('abstract', 'to', '3') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long), [-1]),
                      ('abstract', 'to', 'abstract') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long), [-1])   
                  }
    
    print(ce_confusion_iterative(ce, motifgraph, ['3',0]))
    # edge index dict from gg
    # loop over ce and count found edges
    # problem: What to do if CE can choose 2 nodes?
        # save current result somewhere else
        # loop over children and give as feedback biggest result
        






def generate_colors(num_colors):
    # Define the number of distinct hues to use
    num_hues = num_colors + 1
    # Generate a list of evenly spaced hues
    hues = [i / num_hues for i in range(num_hues)]
    # Shuffle the hues randomly
    #random.shuffle(hues)
    saturations = []
    #saturations = [0.8 for _ in range(num_colors)]
    values = []
    #values = [0.4 for _ in range(num_colors)]
    for i in range(num_colors):
        if i % 2 == 0:
            values.append(0.4)
            saturations.append(0.4)
        else:
            values.append(0.8)
            saturations.append(0.7)
    # Convert the hues, saturations, and values to RGB colors
    colors = [colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hues, saturations, values)]
    # Convert the RGB colors to hexadecimal strings
    hex_colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in colors]
    return hex_colors


# TODO: Visualize every edge-type with a different color
# This function is only used for graphs saved as a dictionary
def visualize(dict_graph):
    graph_final = dgl.heterograph(dict_graph)
    graph_final_hom = dgl.to_homogeneous(graph_final)
    graph_final_nx = dgl.to_networkx(graph_final_hom)
    options = {
        'node_color': 'black',
        'node_size': 20,
        'width': 1,
    }
    plt.figure(figsize=[15, 7])
    nx.draw(graph_final_nx,**options)
    name_plot_save = 'content/plots/graph.pdf'
    name_plot_save = uniquify(name_plot_save)
    #plt.figure()
    plt.savefig(name_plot_save, format="pdf")
    plt.show()
    plt.clf()


# TODO: Use the same colors for all graphs, problem: Sometimes not all node-types are used in the graph
def visualize_heterodata(hd, addname = '', ce = None, gnnout = None, mean_acc = None, list_all_nodetypes = None, label_to_explain = None):
    plt.clf()
    options = {
        'with_labels' : 'True',
        'node_size' : 500
    }
        # create random colours for visualization
    number_of_node_types = len(hd.node_types)
    number_of_node_types_for_colors = number_of_node_types
    
    
    curent_nodetypes_to_all_nodetypes = []
    for _ in range(len(hd.node_types)):
        if list_all_nodetypes != None:
            all_nodetypes_index = list_all_nodetypes.index(hd.node_types[_])
        else:
            all_nodetypes_index = _
        curent_nodetypes_to_all_nodetypes.append([_, all_nodetypes_index])
    
    
    if list_all_nodetypes != None:
        number_of_node_types_for_colors = len(list_all_nodetypes) 
    colors = generate_colors(number_of_node_types_for_colors)
    if number_of_node_types_for_colors == 4:
        colors = ['#59a14f', '#f28e2b', '#4e79a7', '#e15759']
    #find out, which node in homogeneous graph has which type
    homdata = hd.to_homogeneous()
    tensor_with_node_types = homdata.node_type
    #generate networkx graph with the according setting
    Gnew = nx.Graph()
    #add nodes
    num_nodes_of_graph = len(homdata.node_type.tolist())
    Gnew.add_nodes_from(list(range(num_nodes_of_graph)))
    #add edges
    list_edges_start, list_edges_end = homdata.edge_index.tolist()[0], homdata.edge_index.tolist()[1]
    list_edges_for_networkx = list(zip(list_edges_start, list_edges_end))
    Gnew.add_edges_from(list_edges_for_networkx)
    #color nodes
    list_node_types = homdata.node_type.tolist()
    node_labels_to_indices = dict()
    index = 0
    stop = False #the prediction is always done for the first node
    for nodekey in homdata.node_type.tolist():
        if label_to_explain != None:
            if str(curent_nodetypes_to_all_nodetypes[nodekey][1]) == label_to_explain and stop == False:
                node_labels_to_indices.update({index : '*'})
                stop = True
            else:
                node_labels_to_indices.update({index : ''})
        else:
            node_labels_to_indices.update({index : curent_nodetypes_to_all_nodetypes[nodekey][1]})
        index +=1
    color_map_of_nodes = []
    for typeindex in list_node_types:
        color_map_of_nodes.append(colors[curent_nodetypes_to_all_nodetypes[typeindex][1]])
    #plt
    nx.draw(Gnew, node_color=color_map_of_nodes,  **options, labels = node_labels_to_indices)
    #create legend
    patch_list = []
    name_list = []
    for i in range(number_of_node_types):
        patch_list.append(plt.Circle((0, 0), 0.1, fc=colors[curent_nodetypes_to_all_nodetypes[i][1]]))
        name_list.append(hd.node_types[i])
    
    
    #create caption
    caption_text = ''
    caption_size = 18
    if ce != None:
        caption_text += ce
        caption_size -= 4
    if gnnout != None:
        caption_text += '\n'+ ' out: ' + str(gnnout)
        caption_size -= 4
    if mean_acc != None:
        caption_text += ' acc: ' + str(mean_acc)
        caption_size -= 4
    caption_position = (0.5, 0.1)  # Adjust the position as per your requirements
    
    
    #show plt
    name_plot_save = 'content/plots/graph'+addname+'wo_text'+'.pdf'
    name_plot_save = uniquify(name_plot_save)
    plt.savefig(name_plot_save, bbox_inches='tight', format="pdf")
    plt.legend(patch_list, name_list)
    plt.figtext(*caption_position, caption_text, ha='center')#, size = caption_size)
    
    #plt.figure()
    name_plot_save = 'content/plots/graph'+addname+'.pdf'
    name_plot_save = uniquify(name_plot_save)
    plt.savefig(name_plot_save, bbox_inches='tight', format="pdf")
    #plt.show()
    
    
   
    



def visualize_best_results(num_top, saved_list, addname = '', list_all_nodetypes = None, label_to_explain = None):
    num_top = min([num_top, len(saved_list)])
    for i in range(num_top):
        saved_dict_result = saved_list[i][0][0]
        saved_features_result = saved_list[i][0][1]
        ce = None
        gnn_out = None
        mean_acc = None
        try:
            gnn_out = saved_list[i][1]
        except Exception as e:
            print(f"290 Here we skiped the error: {e}")
        try:
            ce = saved_list[i][2]
        except Exception as e:
            print(f"290 Here we skiped the error: {e}")
        try:
            mean_acc = saved_list[i][3]
        except Exception as e:
            print(f"290 Here we skiped the error: {e}")
        visualize_heterodata(graphdict_and_features_to_heterodata(saved_dict_result, saved_features_result), addname, gnnout = gnn_out, ce = ce, mean_acc = mean_acc, list_all_nodetypes = list_all_nodetypes, label_to_explain = label_to_explain)
        '''
        try:
            visualize_heterodata(graphdict_and_features_to_heterodata(saved_dict_result, saved_features_result), addname, gnnout = gnn_out, ce = ce, int_generate_colors = int_generate_colors)
        except Exception as e:
            print(f"290 Here we skiped the error: {e}")
            print(291, i, saved_list[i])
           ''' 

def visualize_best_ces(num_top_ces, num_top_graphs, list_of_results_ce = list(), list_all_nodetypes = None, label_to_explain = None, ml = None, ds = None, node_expl = None, plotname = 'any'):
    #for each CE: visualize num_top_graphs by saving them under a unique name
    #aufbau: [graphs, ce]
    num_top_ces = min(num_top_ces, len(list_of_results_ce))
    for ind_ce in range(num_top_ces):
        if ml != None and ds != None and node_expl != None:
            ce_fid = ce_fidelity(list_of_results_ce[ind_ce][5], modelfid=ml, datasetfid = ds, node_type_expl = node_expl)
            print(378, ce_fid)
        else:
            ce_fid = -1
        ce_vis = list_of_results_ce[ind_ce][1]
        score = list_of_results_ce[ind_ce][2]
        avg_acc = list_of_results_ce[ind_ce][3]
        max_acc = list_of_results_ce[ind_ce][4]
        # new way to compute accuracy directly on CE:
        ce_dict = generate_cedict_from_ce(list_of_results_ce[ind_ce][5])
        #compute_confusion_for_ce_line(ce_dict)
        
        
        print(364, avg_acc, ce_vis, num_top_ces)
        graphs = list_of_results_ce[ind_ce][0]
        addname = plotname+str(ind_ce+1)
        num_top_graphs_local = min(num_top_graphs, len(graphs))
        for ind_gr in range(num_top_graphs_local):
            dict_graph_ce = graphs[ind_gr][0][0]
            features_graph_ce = graphs[ind_gr][0][1]
            gnn_out = None
            try:
                gnn_out = graphs[ind_gr][1]
            except Exception as e:
                print(f"290 Here we skiped the error: {e}")
            visualize_heterodata(graphdict_and_features_to_heterodata(dict_graph_ce, features_graph_ce), addname, gnnout = 'Score: '+str(score), ce = ce_vis, mean_acc = str(avg_acc)+' max acc: ' + str(max_acc) + ' fid: ' + str(ce_fid), list_all_nodetypes = list_all_nodetypes, label_to_explain = label_to_explain)
        
    
    
    
# ------------------  Testing Zone -----------------------    
    
def test_ce_prediction():
    ce_test = create_test_ce()
    graph_test =  {('3', 'to', '2') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long)),
                     ('2', 'to', '3') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long)),
                ('2', 'to', '1') :(torch.tensor([0], dtype = torch.long), 
            torch.tensor([0], dtype = torch.long)),
                     ('1', 'to', '2') :(torch.tensor([0], dtype = torch.long)),
                     }
    print(compute_prediction_ce(ce_dict=ce_test, graph_dict=graph_test, node_type_to_explain='3', index_to_explain=0))
    
   # sys.exit()
#test_ce_prediction()




#compute_confusion_for_ce_line(1) #random input at first



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# ------- Start: Delete old graph-pdfs from content/plots folder
delete_files_in_folder('content/plots')

# ----------------- Evaluate and visualize examples from DBLP Dataset
hd = data
should_new_graphs_be_created = False 
#list_number_of_edges = [2,3,5,7]
#list_percent_new_node = [0.2, 0.5, 0.7, 0.95]
list_number_of_edges = [3]
list_percent_new_node = [0.95]
examples_to_test = 10
target_node_type_to_explain = 'author'
cat_to_explain = 3
filename = "content/created_graphs/"+'DBLP'

'''
saved_list_dblp = create_graphs_for_heterodata(hd, should_new_graphs_be_created,
                                 list_number_of_edges, list_percent_new_node, 
                                 examples_to_test, target_node_type_to_explain, cat_to_explain, filename,
                                 model
                                 )
#visualize_best_results(1, saved_list_dblp)
print(339, 'DBLP Has Worked')
'''


# -------------   create random BA House Graphs, evaluate a GNN on it and generate approximative example graphs
# generate BAHouses (hetero):

bashapes = create_hetero_ba_houses(500,100)
#print('Created BAShapes:', bashapes)
#visualize_heterodata(bashapes)

#create ontology
#from onto import save_ontology_from_hdata
#save_ontology_from_hdata(bashapes, 'BAShapes_onto')


# train GNN
modelHeteroBSM = bsm.train_GNN(True, bashapes)


# random example graphs:
hd = bashapes
should_new_graphs_be_created = True
list_number_of_edges = [2,3,5,7]
list_percent_new_node = [0.2, 0.5, 0.7, 0.95]
examples_to_test = 10
target_node_type_to_explain = '3'
cat_to_explain = 1
filename = "content/created_graphs/"+'HeteroBAShapes'
model = modelHeteroBSM

saved_list_bashapes = create_graphs_for_heterodata(hd, should_new_graphs_be_created,
                                 list_number_of_edges, list_percent_new_node, 
                                 examples_to_test, target_node_type_to_explain, cat_to_explain, filename,
                                 model
                                 )
#visualize_best_results(7, saved_list_bashapes)



# -------------   create random BA House Class Expressions, create graphs from these Class expressions, evaluate a GNN on it

'''
def create_ce_and_graphs_BAHetero(data, model, iterations = 300 ):
    saved_graphs_ces = "content/created_graphs/"+'CEs_for_HeteroBAShapes'
    list_length_of_ce = [4,5,7,8,10]
    result_graphs = list()
    for _ in range(iterations):
        len_ce = random.choice(list_length_of_ce)
        ce = create_random_ce_from_BAHetero(len_ce)
        dict_graph = create_graphdict_from_ce(ce, ['0','1','2','3'], ['to']) #possible nodes, edges have to be transmitted; features can be completed afterwards
        #hd_prev = graphdict_and_features_to_heterodata(dict_graph)
        #visualize_heterodata(hd_prev)
        ce_str = ce
        try: 
            ce_str = dlsr.render(ce)
        except Exception as e:
            print(f"371 Here we skiped the error: {e}")
        #add features
        list_results = list()
        new_graph = add_features_and_predict_outcome(10,1, model,data,list_results,dict_graph,saved_graphs_ces, ce_str = ce_str)
        result_graphs += new_graph
        result_graphs = sorted(result_graphs, key=lambda x: x[1], reverse = True)
    print(376, len(result_graphs))
    visualize_best_results(5, result_graphs, 'ce', ['0','1','2','3'], label_to_explain = '3')
 '''                                            

        
#create_ce_and_graphs_BAHetero(bashapes, modelHeteroBSM)




# -----------------  Write this in functions better usable for future datasets
def ce_score_fct(ce, list_gnn_outs, lambdaone, lambdatwo):
    #avg_gnn_outs-lambda len_ce - lambda_var
    length_of_ce = length_ce(ce)
    mean = sum(list_gnn_outs) / len(list_gnn_outs)
    squared_diffs = [(x - mean) ** 2 for x in list_gnn_outs]
    sum_squared_diffs = sum(squared_diffs)
    variance = sum_squared_diffs / (len(list_gnn_outs))
    return mean-lambdaone*length_of_ce-lambdatwo*variance









def create_ce_and_graphs_to_heterodataset(hd, model, dataset_name, target_class, cat_to_explain = -1, graphs_per_ce = 100, iterations = 300, compute_acc = False):
    saved_graphs_ces = "content/created_graphs/"+'CEs_for_' + dataset_name
    #node types
    node_types = hd.node_types
    metagraph = hd.edge_types #[str,str,str]
    edge_names = []
    for mp in metagraph:
        if mp[1] not in edge_names:
            edge_names.append(mp[1])
    list_length_of_ce = [3,4,5]#,6,7,8]
    result_graphs = list()
    result_graphs_acc = list()
    result_ces = list()
    for _ in range(iterations):
        try:
            len_ce = random.choice(list_length_of_ce)
            ce_here = create_random_ce_from_metagraph(len_ce, metagraph, target_class)
            local_dict_results = dict()
            local_dict_results['acc'] = []
            local_dict_results['GNN_outs'] = []
            for griter in range(graphs_per_ce):
                dict_graph = create_graphdict_from_ce(ce_here, node_types, edge_names, metagraph) #possible nodes, edges have to be transmitted; features can be completed afterwards
                #on all: Calculate gnn and save the outs
                ce_str = ce_here
                try:
                    ce_str = dlsr.render(ce_here)
                except Exception as e:
                    print(f"789 Here we skiped the error: {e}")
                list_results = list()
                result = add_features_and_predict_outcome(1 ,cat_to_explain,  model, hd, list_results, dict_graph, saved_graphs_ces, ce_str = ce_str, compute_acc=compute_acc)
                local_dict_results['GNN_outs'].append(result[0][1])
                # calculate accuracy
                local_dict_results['acc'].append(result[0][3])
            mean_acc = 0
            '''
            for acc in local_dict_results['acc']:
                mean_acc +=float(acc)
            mean_acc = mean_acc/len(local_dict_results['acc'])
            mean_acc = round(mean_acc, 2)
            '''      
            # we need a result: [CE, averaged graph accuracy, CE_score, 1-3 example graphs]
            list_gnn_outs = local_dict_results['GNN_outs']
            lambdaone = 0.04
            lambdatwo = 0.08
            score = ce_score_fct(ce_here, list_gnn_outs, lambdaone, lambdatwo)
            list_accs =  [inner_list[0] for inner_list in local_dict_results['acc']]
            avg_graph_acc = sum(list_accs) / len(list_accs)
            max_graph_acc = max(list_accs)
            
            
            
            
            
            
            
            
            
            
            #print(530, list_accs, avg_graph_acc, len(list_accs), sum(list_accs))
            #idea: create 10 example graphs and save top 3 graphs here
            local_result_ce = [ce_here, score, avg_graph_acc, max_graph_acc]
            result_ces.append(local_result_ce)
            '''
            dict_graph = {('3', 'to', '2') :(torch.tensor([0], dtype = torch.long), 
                torch.tensor([0], dtype = torch.long)),
                         ('2', 'to', '3') :(torch.tensor([0], dtype = torch.long), 
                torch.tensor([0], dtype = torch.long)),
                          ('2', 'to', '2') :(torch.tensor([0,1], dtype = torch.long), 
                torch.tensor([1,0], dtype = torch.long))
                         }
            '''     
            #create one graph for visualization
            ce_str = ce_here
            try: 
                ce_str = dlsr.render(ce_here)
            except Exception as e:
                print(f"838 Here we skiped the error: {e}")
            #add features
            list_results = list()
            new_graph = add_features_and_predict_outcome(1 ,cat_to_explain,  model, hd, list_results, dict_graph, saved_graphs_ces, ce_str = ce_str,compute_acc=compute_acc)
            result_graphs += new_graph
            result_graphs = sorted(result_graphs, key=lambda x: x[1], reverse = True)
        except Exception as e:
                    print(f"845 Here we skiped the error: {e}")
    #visualize_best_results(10, result_graphs, 'ce', node_types, label_to_explain = '3')
    #get 5 best ces and create 10 sample graphs for each of them
    result_ces = sorted(result_ces, key=lambda x: x[1], reverse = True)
    local_graphs_for_ce = list()
    total_graph_to_ce_list = list()
    for ceindex in range(iterations):
        try: 
            local_graphs_for_ce = list()
            cel_best = result_ces[ceindex][0]

            # HERE compute fidelity!

            #a) create a new BA-Shapes Dataset
            #b) compute GNN on test-set
            #c) evaluate CEs on test-set


            ce_score = round(result_ces[ceindex][1],2)
            ce_avg_graph = round(result_ces[ceindex][2],2)
            ce_max_graph = round(result_ces[ceindex][3],2)
            cel_str = cel_best
            try: 
                cel_str = dlsr.render(cel_best)
            except Exception as e:
                print(f"870 Here we skiped the error: {e}")
            dict_graph = dict()
            new_graph = list()
            for _ in range(16):
                dict_graph = create_graphdict_from_ce(cel_best, node_types, edge_names, metagraph)
                new_graph = add_features_and_predict_outcome(1 ,cat_to_explain,  model, hd, list(), dict_graph, saved_graphs_ces, ce_str = cel_str,compute_acc=compute_acc)
                local_graphs_for_ce += new_graph
            local_graphs_for_ce = sorted(local_graphs_for_ce, key=lambda x: x[1], reverse = True)
            total_graph_to_ce_list.append([local_graphs_for_ce, cel_str,ce_score, ce_avg_graph, ce_max_graph, cel_best])
            print(576, cel_str,ce_score, ce_avg_graph, ce_max_graph)
        except Exception as e:
                    print(f"881 Here we skiped the error: {e}")
    total_graph_to_ce_list = sorted(total_graph_to_ce_list, key=lambda x: x[2], reverse = True)
    visualize_best_ces(num_top_ces = 6, num_top_graphs = 3, list_of_results_ce = total_graph_to_ce_list, list_all_nodetypes = node_types, label_to_explain = target_class, ml=model, ds=hd, node_expl = target_class, plotname = dataset_name)
    #visualize results in an own function using total_graph_to_ce_list
    
        
    

    
    
#create_ce_and_graphs_to_heterodataset(datadblp, modeldblp, 'DBLP', 'Author', cat_to_explain = -1,  graphs_per_ce = 5, iterations = 50,compute_acc=False)
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# ----------- testing
ce_test = create_test_ce()[1] #3-1-2
#print(919, dlsr.render(ce_test))
#ce_confusion(ce_test, motif = 'house') #should output 0
    
    
    
    
#---------------------- evaluation DBLP 
# this does not work accordingly
#create_ce_and_graphs_to_heterodataset(datadblp, modeldblp, 'DBLP', 'author', cat_to_explain = -1,  graphs_per_ce = 8, iterations = 50,compute_acc=False)
    
    
# ---------------------- evaluation HeteroBAShapes
    
    
create_ce_and_graphs_to_heterodataset(bashapes, modelHeteroBSM, 'HeteroBAShapes', '3', cat_to_explain = -1,  graphs_per_ce = 16, iterations = 1000, compute_acc=True)



































# --------------------------------- test

def test_ce_and_graph_creation(hd, model, dataset_name, target_class, cat_to_explain = -1):
    
    
    node_types = hd.node_types
    metagraph = hd.edge_types #[str,str,str]
    edge_names = []
    
    for mp in metagraph:
        if mp[1] not in edge_names:
            edge_names.append(mp[1])
    ce_test = create_random_ce_from_metagraph(6, metagraph, target_class)
    print(611, dlsr.render(ce_test))
    dict_graph = create_graphdict_from_ce(ce_test, node_types, edge_names, metagraph)
    print(619, dict_graph)
#for _ in range(7):
#    test_ce_and_graph_creation(bashapes, modelHeteroBSM, 'HeteroBAShapes', '3', cat_to_explain = -1)
    

#dblp


# not working yet, but maybe will one day.
target_category_DBLP = 'conference'
dataset = DBLP(path, transform=T.Constant(node_types=target_category_DBLP))
target_category_DBLP = 'author'  # we want to predict classes of author
# 4 different classes for author:
#   database, data mining, machine learning, information retrieval.
data = dataset[0]
model = HeteroGNN(data.metadata(), hidden_channels=32, out_channels=4,
                  num_layers=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)



target_node_type_to_explain = 'author'
cat_to_explain = 3
filename = "content/created_graphs/"+'DBLP'



#create_ce_and_graphs_to_heterodataset(data, model, 'DBLP', target_class = target_node_type_to_explain, cat_to_explain = cat_to_explain, iterations = 2)