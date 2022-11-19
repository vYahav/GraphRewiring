import torch
import torch_geometric as tg
from torch_geometric.datasets import Planetoid
from matplotlib import pyplot as plt
import sys
import numpy as np
import copy 
from collections import OrderedDict
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GAT(torch.nn.Module):
    def __init__(self,num_of_features,num_of_classes):
        super(GAT, self).__init__()
        self.num_of_features    = num_of_features
        self.num_of_classes     = num_of_classes
        self.hid                = 8
        self.in_head            = 8
        self.out_head           = 1

        self.conv1 = tg.nn.GATConv(self.num_of_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = tg.nn.GATConv(self.hid * self.in_head, self.num_of_classes, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, data,edges = None):
        x, edge_index = data.x, data.edge_index

        if edges is not None:
            first_edge_index = edges[0]
            second_edge_index = edges[1]
        else:
            first_edge_index = edge_index
            second_edge_index = edge_index

        # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
        # One can skip them if the dataset is sufficiently large.

        x = torch.nn.functional.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x,first_edge_index )
        x = torch.nn.functional.elu(x)
        x = torch.nn.functional.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, second_edge_index)

        return torch.nn.functional.log_softmax(x, dim=1)



def train_GAT(model,data,graphs= None,verbose=True,return_final_acc=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    data  = data.to(device)

    optimizer   = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    
    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        out = model(data,graphs)
        loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])

        if epoch % 200 == 0 and verbose:
            print(loss)

        loss.backward()
        optimizer.step()
    
    model.eval()
    _, pred = model(data,graphs).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    if verbose:
        print('Accuracy: {:.4f}'.format(acc))

    if return_final_acc:
        return model,acc

    else:
        return model

def rewire_graph(model,graph,number_of_nodes,remove_ratio=0.1,add_ratio=0.1,add_to_each_node=False,last=False, matrix_scores=None,return_matrix_scores=False,graph_edges_to_mod=None):
    model.to(device)
    graph.to(device)
    FC_edges = torch.combinations(torch.tensor([i for i in range(number_of_nodes)])).to(device)
    
    first_attn_layer    = model.get_submodule('conv1')
    second_attn_layer    = model.get_submodule('conv2')

    if matrix_scores is None:
        if(last == True):
            x = first_attn_layer(graph.x,graph.edge_index)
            x = torch.nn.functional.elu(x)

            _,attn_weights      = second_attn_layer(x,FC_edges.T,return_attention_weights=True)
        else:
            _,attn_weights      = first_attn_layer(graph.x,FC_edges.T,return_attention_weights=True)
        


        edges, scores       = attn_weights
        mean_scores = scores.mean(dim=1).cpu().detach().numpy()
        matrix_scores = edge_list_to_matrix_scores(edges,mean_scores,number_of_nodes,normalized=True,allow_self_loops=False)

    #edges to remove
    if graph_edges_to_mod is not None:
        filtered_edges = prune_edges_using_scores(graph_edges_to_mod,matrix_scores,remove_ratio=remove_ratio)
    else:
        filtered_edges = prune_edges_using_scores(graph.edge_index,matrix_scores,remove_ratio=remove_ratio)

    #edges to add
    if add_to_each_node and add_ratio > 0:
        edges_to_add = add_top_edge_for_each_node(matrix_scores)
    else:
        edges_to_add = add_edges_using_scores(graph.edge_index,matrix_scores,number_of_nodes,add_ratio)

    if(edges_to_add is not None):
        new_edges = torch.hstack([filtered_edges,edges_to_add.to(device)])
    else:
        new_edges = filtered_edges

    new_edges = tg.utils.remove_self_loops(new_edges)[0]
    
    new_graph = copy.copy(graph)
    new_graph.edge_index = new_edges

    if return_matrix_scores:
        return new_graph, matrix_scores
    else:
        return new_graph 

def get_edges_scores(edge_list,matrix_scores):
    edges_scores = []
    for edge in edge_list.T:
        score = matrix_scores[edge[0],edge[1]]
        edges_scores.append(score)
    return torch.hstack(edges_scores)

def add_top_edge_for_each_node(matrix_scores):
    top_edges = torch.vstack([torch.tensor(range(matrix_scores.shape[0]),device=device),matrix_scores.argmax(dim=1).to(device)])
    return top_edges

def add_edges_using_scores(edge_list,matrix_scores,number_of_nodes,add_ratio=0.1):
    matrix_score_mod = copy.copy(matrix_scores)
    number_of_edges     = edge_list.shape[1]
    number_of_new_edges = int(number_of_edges*(add_ratio))
    new_edges = []
    for i in range(number_of_new_edges):
        new_edge_index  = matrix_score_mod.argmax()
        new_edge_index = torch.tensor([new_edge_index//number_of_nodes,new_edge_index%number_of_nodes])
        matrix_score_mod[new_edge_index] = -1
        new_edges.append(new_edge_index)
    if len(new_edges) >0:
        new_edges   =  torch.vstack(new_edges)
        return new_edges.T
    else:
        return None
def prune_edges_using_scores(edge_list,matrix_scores,remove_ratio=0.1):
    edges_scores        = get_edges_scores(edge_list,matrix_scores)
    number_of_edges     = edges_scores.shape[0]
    pruned_num_of_edges = int(number_of_edges*(1-remove_ratio))
    new_edges_index     = edges_scores.topk(pruned_num_of_edges)[1]
    pruned_edges        = edge_list.T[new_edges_index].T
    return pruned_edges


def filter_edge_list_by_matrix_score(edge_list,matrix_scores,threshold_value):
    filtered_edges_list = []
    for edge in edge_list.T:
        edge_score = matrix_scores[edge[0],edge[1]]
        if edge_score >= threshold_value:
            filtered_edges_list.append(edge)
        
    filtered_edges = torch.vstack(filtered_edges_list)
    return filtered_edges



def edge_list_to_matrix_scores(edge_list,attention_scores,num_of_nodes,normalized=True,allow_self_loops=False):

    attention_matrix = torch.zeros((num_of_nodes,num_of_nodes))
    attention_scores = torch.tensor(attention_scores)
    attention_matrix[tuple(edge_list[0]),tuple(edge_list[1])] = attention_scores

    if allow_self_loops == False:
        attention_matrix.fill_diagonal_(0)

    if normalized:
        attention_matrix = torch.nn.functional.softmax(attention_matrix,dim=1)
    
    return attention_matrix



def load_dataset():
    name_data = 'Cora'
    dataset = Planetoid(root= '/tmp/' + name_data, name = name_data)
    dataset.transform = tg.transforms.NormalizeFeatures()
    
    print(f"Number of Classes in {name_data}:", dataset.num_classes)
    print(f"Number of Node Features in {name_data}:", dataset.num_node_features)

    return dataset



def run_tests(remove_ratio=0.1,add_ratio=0.1,add_to_each_node=False):
    dataset = load_dataset()
    graph_data = dataset[0]
    results_dict = {}

    print(f'Rewire_manager > Running Experiments with remove_ratio = {remove_ratio} and add_ratio = {add_ratio}')

    model = GAT(dataset.num_node_features,dataset.num_classes)

    print('Rewire_Manager > Training base model')
    model,base_model_acc = train_GAT(model,graph_data,return_final_acc=True,verbose=False)
    results_dict['Base_model'] = base_model_acc
    #prepare all graphs

    #graphs for first layer
    new_graph_first_only_prune, matrix_scores   = rewire_graph(model,graph_data,graph_data.x.shape[0],remove_ratio=remove_ratio,add_ratio=0,add_to_each_node=False,last=False,matrix_scores=None,return_matrix_scores=True)
    new_graph_first_only_add                    = rewire_graph(model,graph_data,graph_data.x.shape[0],remove_ratio=0,add_ratio=add_ratio,add_to_each_node=add_to_each_node,last=False,matrix_scores=matrix_scores)
    new_graph_first_add_and_prune               = rewire_graph(model,graph_data,graph_data.x.shape[0],remove_ratio=remove_ratio,add_ratio=add_ratio,add_to_each_node=add_to_each_node,last=False,matrix_scores=matrix_scores)

    first_layer_tests = OrderedDict({
        'First_Layer > Prune'           : new_graph_first_only_prune,
        'First_Layer > Add'             : new_graph_first_only_add,
        'First_Layer > Prune And Add'   : new_graph_first_add_and_prune,
    })

    #graphs for second layer
    new_graph_second_only_prune,matrix_scores   = rewire_graph(model,graph_data,graph_data.x.shape[0],remove_ratio=remove_ratio,add_ratio=0,add_to_each_node=False,last=True,matrix_scores=None,return_matrix_scores=True)
    new_graph_second_only_add                   = rewire_graph(model,graph_data,graph_data.x.shape[0],remove_ratio=0,add_ratio=add_ratio,add_to_each_node=add_to_each_node,last=True,matrix_scores=matrix_scores)
    new_graph_second_add_and_prune              = rewire_graph(model,graph_data,graph_data.x.shape[0],remove_ratio=remove_ratio,add_ratio=add_ratio,add_to_each_node=add_to_each_node,last=True,matrix_scores=matrix_scores)

    second_layer_tests = OrderedDict({
        'Second_Layer > Prune'           : new_graph_second_only_prune,
        'Second_Layer > Add'             : new_graph_second_only_add,
        'Second_Layer > Prune And Add'   : new_graph_second_add_and_prune,
    })
    print(f'Base_model_acc = {base_model_acc}')
    #first layer only tests
    print("------First Layer Rewiring Tests-------")
    for test_name,test_graph in first_layer_tests.items():
        test_model = GAT(dataset.num_node_features,dataset.num_classes)
        test_model,test_acc = train_GAT(test_model,test_graph,return_final_acc=True,verbose=False)
        print(f'{test_name} - {test_acc}')
        results_dict[test_name] = test_acc

    #Second layer only tests
    print("------Second Layer Rewiring Tests-------")
    for test_name,test_graph in second_layer_tests.items():
        test_model = GAT(dataset.num_node_features,dataset.num_classes)
        test_model,test_acc = train_GAT(test_model,test_graph,return_final_acc=True,verbose=False)
        print(f'{test_name} - {test_acc}')
        results_dict[test_name] = test_acc

    #Second layer only tests
    print("------Both Layer Rewiring Tests-------")
    for test_name,first_layer_graph in first_layer_tests.items():
        test_model = GAT(dataset.num_node_features,dataset.num_classes)
        second_layer_graph = second_layer_tests[test_name.replace('First','Second')]
        test_model,test_acc = train_GAT(test_model,test_graph,graphs = (first_layer_graph.edge_index,second_layer_graph.edge_index),return_final_acc=True,verbose=False)
        results_dict[test_name.replace('First_Layer','Both_Layers')] = test_acc

    print(f'---------Experiment Results (remove_ratio = {remove_ratio} | add_ratio = {add_ratio} -----------')

    for test_name,test_acc in results_dict.items():
        print(f'{test_name} - {test_acc}')

    print('-------------------------------------')

    return results_dict



def test_iterative_rewire():
    print('Running iterative Method')
    dataset = load_dataset()
    graph_data = dataset[0]
    results_dict = OrderedDict()
    remove_ratio    = 0.003
    add_ratio       = 0.0025
    number_of_iterations = 100
    itr_acc_list = []


    print('Beggining iterations')
    model = GAT(dataset.num_node_features,dataset.num_classes)
    model,base_model_acc = train_GAT(model,graph_data,return_final_acc=True,verbose=False)
    results_dict['iter_0'] = base_model_acc
    itr_acc_list.append(base_model_acc)
    print(f'iter_0 - base_model_acc {base_model_acc}')
    graph_first_layer   = copy.copy(graph_data)
    graph_second_layer  = copy.copy(graph_data)
    for iter_num in tqdm(range(1,number_of_iterations+1)):
        graph_first_layer   = rewire_graph(model,graph_first_layer,graph_data.x.shape[0],remove_ratio=remove_ratio,add_ratio=add_ratio,add_to_each_node=False,last=False,matrix_scores=None)
        graph_second_layer  = rewire_graph(model,graph_first_layer,graph_data.x.shape[0],remove_ratio=remove_ratio,add_ratio=add_ratio,add_to_each_node=False,last=True,matrix_scores=None,graph_edges_to_mod=graph_second_layer.edge_index)
        iter_model =  GAT(dataset.num_node_features,dataset.num_classes)
        iter_model,iter_acc = train_GAT(iter_model,graph_data,graphs = (graph_first_layer.edge_index,graph_second_layer.edge_index),return_final_acc=True,verbose=False)
        itr_name = 'iter_' + str(iter_num)
        results_dict[itr_name] = iter_acc
        itr_acc_list.append(iter_acc)
        print(f'itr {iter_num} - {str(iter_acc)} - number_of_edges = {graph_first_layer.edge_index.shape[1]}')
        remove_ratio    = remove_ratio/1.1
        add_ratio       =   add_ratio/1.1

        if iter_num != 0 and iter_num%10 == 0:
            plt.plot(itr_acc_list)
            plt.title('Accuracy for Iteration')
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy')
            plt.show()


    for itr_name,itr_acc in results_dict.items():
        print(f'{itr_name} - {itr_acc}')
    
    plt.plot(itr_acc_list)
    plt.title('Accuracy for Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()



if __name__ == '__main__':

    if (len(sys.argv) == 2) and sys.argv[1] == 'iterative':
        test_iterative_rewire()
        exit(0)

    ratio_list = [0.005,0.01,0.015,0.02]

    RUN_ADD = False


    done_list = [(0.005,0.005),(0.005,0.01)]

    for add_ratio in ratio_list:
        for remove_ratio in ratio_list:
            if (add_ratio,remove_ratio) in done_list:
                continue

            print('Running Experiments using precentage ADD')
            final_results_dict_per = {}
            number_of_runs = 10
            print(f'Running {number_of_runs} tests and accumulating results')
            for i in tqdm(range(number_of_runs)):
                results = run_tests(remove_ratio=remove_ratio,add_ratio=add_ratio,add_to_each_node=False)
                for test_name,test_result in results.items():
                    if test_name in final_results_dict_per:
                        final_results_dict_per[test_name].append(test_result)
                    else:
                        final_results_dict_per[test_name] = [test_result]

            for test_name,test_result_list in final_results_dict_per.items():
                print(f'{test_name} percentage - avg_acc = {np.mean(test_result_list)} | max_acc = {np.max(test_result_list)} | std = {np.std(test_result_list)}')
            
                    
            if RUN_ADD:
                print('Running Experiments using each node ADD')
                final_results_dict_each_node = {}
                number_of_runs = 10
                print(f'Running {number_of_runs} tests and accumulating results')
                for i in tqdm(range(number_of_runs)):
                    results = run_tests(remove_ratio=remove_ratio,add_ratio=add_ratio,add_to_each_node=True)
                    for test_name,test_result in results.items():
                        if test_name in final_results_dict_each_node:
                            final_results_dict_each_node[test_name].append(test_result)
                        else:
                            final_results_dict_each_node[test_name] = [test_result]

                for test_name,test_result_list in final_results_dict_each_node.items():
                    print(f'{test_name} each-node - avg_acc = {np.mean(test_result_list)} | max_acc = {np.max(test_result_list)} | std = {np.std(test_result_list)}')
            

            with(open(f'percetage_results_{str(add_ratio)}_{str(remove_ratio)}.txt','w')) as fl:
                for test_name,test_result_list in final_results_dict_per.items():
                    fl.write(f'{test_name} - avg_acc = {np.mean(test_result_list)} | max_acc = {np.max(test_result_list)} | std = {np.std(test_result_list)} \n')

            if RUN_ADD:
                with(open(f'each_node_results_{str(add_ratio)}_{str(remove_ratio)}.txt','w')) as fl:
                    for test_name,test_result_list in final_results_dict_each_node.items():
                        fl.write(f'{test_name} - avg_acc = {np.mean(test_result_list)} | max_acc = {np.max(test_result_list)} | std = {np.std(test_result_list)} \n')

            with(open(f'percetage_results_full_{str(add_ratio)}_{str(remove_ratio)}.txt','w')) as fl:
                for test_name,test_result_list in final_results_dict_per.items():
                    for test_result in test_result_list:
                        fl.write(f'{test_name} - acc = {test_result} \n')

            if RUN_ADD:
                with(open(f'each_node_results_full_{str(add_ratio)}_{str(remove_ratio)}.txt','w')) as fl:
                    for test_name,test_result_list in final_results_dict_each_node.items():
                        for test_result in test_result_list:
                            fl.write(f'{test_name} - acc = {test_result} \n')
