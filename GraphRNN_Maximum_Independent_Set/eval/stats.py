import concurrent.futures
from datetime import datetime
from functools import partial
import numpy as np
import networkx as nx
import os
import pickle as pkl
import subprocess as sp
import time
from eval.bilaketaHeuristikoak import simulatedAnnealing_deia, EDA_deia, funtzioEraikitzaile_deia
import args
from itertools import zip_longest


args = args.Args()
import eval.mmd as mmd

PRINT_TIME = False

def degree_worker(G):
    return np.array(nx.degree_histogram(G))

def add_tensor(x,y):
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    return x+y

def degree_stats(mmdTrainTest, degreeTrain, degreeTest, graph_pred_list, is_parallel=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = degreeTrain
    sample_ref2 = degreeTest
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:

        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    mmd_dist_ref_pred = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd)
    mmd_dist_ref2_pred = mmd.compute_mmd(sample_ref2, sample_pred, kernel=mmd.gaussian_emd)
    mmd_dist_ref_ref2 = mmdTrainTest
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return [mmd_dist_ref_pred, mmd_dist_ref2_pred, mmd_dist_ref_ref2,sample_ref, sample_ref2, sample_pred]


def degree_statsTrainTest(graph_ref_list,graph_ref2_list, is_parallel=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_ref2 = []

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref2_list):
                sample_ref2.append(deg_hist)
 

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_ref2_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref2_list[i]))
            sample_ref2.append(degree_temp)


    mmd_dist_ref_ref2 = mmd.compute_mmd(sample_ref, sample_ref2, kernel=mmd.gaussian_emd)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return [mmd_dist_ref_ref2,sample_ref, sample_ref2]



        
def adjacency_matrix_worker(graph):
    """Compute the adjacency matrix of a graph."""
    return nx.to_numpy_array(graph)

def LOP_helburu_funtzioa(X):
    helb_fn = 0
    for i in range(1, X.shape[0]): 
        for j in range(i, X.shape[0]):
            helb_fn += X[i][j] - X[j][i]
    return helb_fn / (X.shape[0]**2)

def gaussian_rbf_kernelLOP(X_list, Y_list, sigma=1.0):
    """Gaussian RBF kernel for adjacency matrices."""
    n = len(X_list)
    m = len(Y_list)
    X_helburu_funtzioa = np.array([LOP_helburu_funtzioa(X) for X in X_list])
    Y_helburu_funtzioa = np.array([LOP_helburu_funtzioa(Y) for Y in Y_list])
    dist = np.abs(X_helburu_funtzioa[:, np.newaxis] - Y_helburu_funtzioa[np.newaxis, :])
    kernel = np.exp(-dist**2 / (2 * sigma**2))
    return kernel.mean()

def mmd_adj_matrices(sample_ref, sample_pred, kernel):
    """Compute MMD between two sets of adjacency matrices using the provided kernel."""
    XX = kernel(sample_ref, sample_ref)
    YY = kernel(sample_pred, sample_pred)
    XY = kernel(sample_ref, sample_pred)
    mmd = XX + YY - 2 * XY
    return mmd


def adjacency_matrix_stats(graph_ref_list, graph_pred_list, is_parallel=False):
    ''' Compute the distance between the adjacency matrices of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_pred_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            sample_ref = list(executor.map(adjacency_matrix_worker, graph_ref_list))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            sample_pred = list(executor.map(adjacency_matrix_worker, graph_pred_list_remove_empty))
    else:
        for i in range(len(graph_ref_list)):
            adj_matrix = nx.to_numpy_array(graph_ref_list[i])
            sample_ref.append(adj_matrix)
        for i in range(len(graph_pred_list_remove_empty)):
            adj_matrix = nx.to_numpy_array(graph_pred_list_remove_empty[i])
            sample_pred.append(adj_matrix)

    print(len(sample_ref), len(sample_pred))
    mmd_dist = mmd_adj_matrices(sample_ref, sample_pred, kernel=gaussian_rbf_kernelLOP)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing adjacency matrix MMD: ', elapsed)
    return mmd_dist





# soluzioak ebazteko hainbat ebatzaile erabiliko dira. Azken finean, ebatzaile bakoitzak soluzioak modu batean lortuko ditu
def emaitzakLortuEbatzaileekin(sample_ref = None, sample_ref2 = None, sample_pred = None, kernel1 = EDA_deia, kernel2 = simulatedAnnealing_deia, kernel3 = funtzioEraikitzaile_deia):
    """Compute MMD between two sets of adjacency matrices using the provided kernel."""
    if kernel1 == EDA_deia:
        argumentuak1 = args.EDA_args
    elif kernel1 == simulatedAnnealing_deia:
        argumentuak1 = args.simulatedAnnealing_args
    elif kernel1 == funtzioEraikitzaile_deia:
        argumentuak1 = args.funtzioEraikitzaile_args

    if kernel2 == EDA_deia:
        argumentuak2 = args.EDA_args
    elif kernel2 == simulatedAnnealing_deia:
        argumentuak2 = args.simulatedAnnealing_args
    elif kernel2 == funtzioEraikitzaile_deia:
        argumentuak2 = args.funtzioEraikitzaile_args

    if kernel3 == EDA_deia:
        argumentuak3 = args.EDA_args
    elif kernel3 == simulatedAnnealing_deia:
        argumentuak3 = args.simulatedAnnealing_args
    elif kernel3 == funtzioEraikitzaile_deia:
        argumentuak3 = args.funtzioEraikitzaile_args


    if sample_ref is not None:
        refList1 = [kernel1(argumentuak1, x) for x in sample_ref]
        print("refList1 done", len(sample_ref))
        refList2 = [kernel2(argumentuak2,x) for x in sample_ref]
        print("refList2 done")
        refList3 = [kernel3(argumentuak3, x) for x in sample_ref]
        print("refList1 done")
    else:
        refList1 = []
        refList2 = []
        refList3 = []

    if sample_ref2 is not None:
        ref2List1 = [kernel1(argumentuak1, x) for x in sample_ref2]
        print("ref2List1 done")
        ref2List2 = [kernel2(argumentuak2,x) for x in sample_ref2]
        print("ref2List2 done")
        ref2List3 = [kernel3(argumentuak3, x) for x in sample_ref2]
        print("ref2List1 done")
    else:
        ref2List1 = []
        ref2List2 = []
        ref2List3 = []

    
    

    if sample_pred is not None:
        predList1 = [kernel1(argumentuak1,x) for x in sample_pred]
        print("predList1 done")
        predList2 = [kernel2(argumentuak2,x) for x in sample_pred]
        print("predList1 done")
        predList3 = [kernel3(argumentuak3,x) for x in sample_pred]
        print("predList1 done")
    else:
        predList1 = []
        predList2 = []
        predList3 = []
    
    

    return refList1, refList2,  refList3, ref2List1, ref2List2, ref2List3, predList1, predList2, predList3


# funtzio honek bi MIS problema multzoren soluzioak itzuliko ditu, ondoren helburuan haien histogramak, edo erabakitzen den metrika aplikatu ditzan
def MIS_value_stats(graph_ref_list, graph_ref2_list, TraintestMMDs, graph_pred_list, is_parallel=False):
    
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    #if is_parallel:
    #    with concurrent.futures.ProcessPoolExecutor() as executor:
    #        sample_ref = list(executor.map(adjacency_matrix_worker, graph_ref_list))
    #    with concurrent.futures.ProcessPoolExecutor() as executor:
    #        sample_pred = list(executor.map(adjacency_matrix_worker, graph_pred_list_remove_empty))
    #else:
    #    for i in range(len(graph_ref_list)):
    #        adj_matrix = nx.to_numpy_array(graph_ref_list[i])
    #        sample_ref.append(adj_matrix)
    #    for i in range(len(graph_pred_list_remove_empty)):
    #        adj_matrix = nx.to_numpy_array(graph_pred_list_remove_empty[i])
    #        sample_pred.append(adj_matrix)
            
    refList1, refList2, refList3 = graph_ref_list
    ref2List1, ref2List2, ref2List3 = graph_ref2_list
    List1_mmd_dist_ref_ref2, List2_mmd_dist_ref_ref2, List3_mmd_dist_ref_ref2 = TraintestMMDs
    
    sample_pred = graph_pred_list
    print(len(sample_pred))
    _,_,_,_,_,_,predList1, predList2, predList3= emaitzakLortuEbatzaileekin(sample_pred = sample_pred, kernel1=simulatedAnnealing_deia, kernel2 = EDA_deia, kernel3 = funtzioEraikitzaile_deia)
    List1_mmd_dist_ref_pred = mmd.compute_mmd([refList1], [predList1], kernel=mmd.gaussian_emd)
    List1_mmd_dist_ref2_pred = mmd.compute_mmd([ref2List1], [predList1], kernel=mmd.gaussian_emd)
    

    List2_mmd_dist_ref_pred = mmd.compute_mmd([refList2], [predList2], kernel=mmd.gaussian_emd)
    List2_mmd_dist_ref2_pred = mmd.compute_mmd([ref2List2], [predList2], kernel=mmd.gaussian_emd)    
    

    List3_mmd_dist_ref_pred = mmd.compute_mmd([refList3], [predList3], kernel=mmd.gaussian_emd)
    List3_mmd_dist_ref2_pred = mmd.compute_mmd([ref2List3], [predList3], kernel=mmd.gaussian_emd)    
    

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing adjacency matrix MMD: ', elapsed)
    return [List1_mmd_dist_ref_pred,List2_mmd_dist_ref_pred,List3_mmd_dist_ref_pred,List1_mmd_dist_ref2_pred,List2_mmd_dist_ref2_pred, List3_mmd_dist_ref2_pred,List1_mmd_dist_ref_ref2, List2_mmd_dist_ref_ref2,List3_mmd_dist_ref_ref2, refList1, refList2, refList3, ref2List1, ref2List2,ref2List3, predList1, predList2, predList3]


# funtzio honek bi MIS problema multzoren soluzioak itzuliko ditu, ondoren helburuan haien histogramak, edo erabakitzen den metrika aplikatu ditzan
def MIS_value_statsTrainTest(graph_ref_list, graph_ref2_list, is_parallel=False):
    sample_ref = []
    sample_ref2 = []
    

    prev = datetime.now()
    #if is_parallel:
    #    with concurrent.futures.ProcessPoolExecutor() as executor:
    #        sample_ref = list(executor.map(adjacency_matrix_worker, graph_ref_list))
    #    with concurrent.futures.ProcessPoolExecutor() as executor:
    #        sample_pred = list(executor.map(adjacency_matrix_worker, graph_pred_list_remove_empty))
    #else:
    #    for i in range(len(graph_ref_list)):
    #        adj_matrix = nx.to_numpy_array(graph_ref_list[i])
    #        sample_ref.append(adj_matrix)
    #    for i in range(len(graph_pred_list_remove_empty)):
    #        adj_matrix = nx.to_numpy_array(graph_pred_list_remove_empty[i])
    #        sample_pred.append(adj_matrix)
            
    sample_ref = graph_ref_list
    sample_ref2 = graph_ref2_list
    refList1, refList2, refList3, ref2List1, ref2List2, ref2List3, _, _, _ = emaitzakLortuEbatzaileekin(sample_ref= sample_ref, sample_ref2= sample_ref2, kernel1=simulatedAnnealing_deia, kernel2 = EDA_deia, kernel3 = funtzioEraikitzaile_deia)
    
    List1_mmd_dist_ref_ref2 = mmd.compute_mmd([refList1], [ref2List1], kernel=mmd.gaussian_emd)
    List2_mmd_dist_ref_ref2 = mmd.compute_mmd([refList2], [ref2List2], kernel=mmd.gaussian_emd)
    List3_mmd_dist_ref_ref2 = mmd.compute_mmd([refList3], [ref2List3], kernel=mmd.gaussian_emd)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing adjacency matrix MMD: ', elapsed)
    return [refList1, refList2, refList3], [ref2List1, ref2List2, ref2List3], [List1_mmd_dist_ref_ref2, List2_mmd_dist_ref_ref2, List3_mmd_dist_ref_ref2]









def emaitzakLortuEbatzaileekin2(sample_ref, sample_pred, kernel1, kernel2):
    """Compute MMD between two sets of adjacency matrices using the provided kernel."""
    if kernel1 == EDA_deia:
        argumentuak1 = args.EDA_args
    elif kernel1 == simulatedAnnealing_deia:
        argumentuak1 = args.simulatedAnnealing_args

    if kernel2 == EDA_deia:
        argumentuak2 = args.EDA_args
    elif kernel2 == simulatedAnnealing_deia:
        argumentuak2 = args.simulatedAnnealing_args

    refList1 = [kernel1(argumentuak1, x) for x in sample_ref]
    print("refList1 done")
    refList2 = [kernel2(argumentuak2,x) for x in sample_ref]
    print("refList2 done")


    return refList1, refList2


# funtzio honek bi MIS problema multzoren soluzioak itzuliko ditu, ondoren helburuan haien histogramak, edo erabakitzen den metrika aplikatu ditzan
def MIS_value_stats2(graph_ref_list, graph_pred_list, is_parallel=False):
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    #if is_parallel:
    #    with concurrent.futures.ProcessPoolExecutor() as executor:
    #        sample_ref = list(executor.map(adjacency_matrix_worker, graph_ref_list))
    #    with concurrent.futures.ProcessPoolExecutor() as executor:
    #        sample_pred = list(executor.map(adjacency_matrix_worker, graph_pred_list_remove_empty))
    #else:
    #    for i in range(len(graph_ref_list)):
    #        adj_matrix = nx.to_numpy_array(graph_ref_list[i])
    #        sample_ref.append(adj_matrix)
    #    for i in range(len(graph_pred_list_remove_empty)):
    #        adj_matrix = nx.to_numpy_array(graph_pred_list_remove_empty[i])
    #        sample_pred.append(adj_matrix)
            
    sample_ref = graph_ref_list
    sample_pred = graph_pred_list
    print(len(sample_ref), len(sample_pred))
    refList1, refList2 = emaitzakLortuEbatzaileekin2(sample_ref, sample_pred, kernel1=simulatedAnnealing_deia, kernel2 = EDA_deia)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing adjacency matrix MMD: ', elapsed)
    return [refList1, refList2, refList1, refList2]

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
            clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist

def clustering_stats(mmdTrainTest, clusteringTrain, clsuteringTest, graph_pred_list, bins=100, is_parallel=True):
    sample_ref = clusteringTrain
    sample_ref2 = clsuteringTest
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, 
                    [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
        # check non-zero elements in hist
        #total = 0
        #for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        #print(total)
    else:
        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                    clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    mmd_dist_ref_pred = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd,
                               sigma=1.0/10, distance_scaling=bins)
    mmd_dist_ref2_pred = mmd.compute_mmd(sample_ref2, sample_pred, kernel=mmd.gaussian_emd,
                               sigma=1.0/10, distance_scaling=bins)
    mmd_dist_ref_ref2 = mmdTrainTest

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return [mmd_dist_ref_pred, mmd_dist_ref2_pred,mmd_dist_ref_ref2,sample_ref,sample_ref2,sample_pred]



def clustering_statsTrainTest(graph_ref_list, graph_ref2_list, bins=100, is_parallel=True):
    sample_ref = []
    sample_ref2 = []

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, 
                    [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, 
                    [(G, bins) for G in graph_ref2_list]):
                sample_ref2.append(clustering_hist)

        # check non-zero elements in hist
        #total = 0
        #for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        #print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                    clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_ref2_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref2_list[i]).values())
            hist, _ = np.histogram(
                    clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref2.append(hist)

 
    

    mmd_dist_ref_ref2 = mmd.compute_mmd(sample_ref, sample_ref2, kernel=mmd.gaussian_emd,
                               sigma=1.0/10, distance_scaling=bins)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return [mmd_dist_ref_ref2,sample_ref,sample_ref2]




# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
        '3path' : [1, 2],
        '4cycle' : [8],
}
COUNT_START_STR = 'orbit counts: \n'

def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges

def orca(graph):
    tmp_fname = 'eval/orca/tmp.txt'
    f = open(tmp_fname, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    output = sp.check_output(['./eval/orca/orca', 'node', '4', 'eval/orca/tmp.txt', 'std'])
    output = output.decode('utf8').strip()
    
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ') ))
          for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts
    

def motif_stats(graph_ref_list, graph_pred_list, motif_type='4cycle', ground_truth_match=None, bins=100):
    # graph motif counts (int for each graph)
    # normalized by graph size
    total_counts_ref = []
    total_counts_pred = []

    num_matches_ref = []
    num_matches_pred = []

    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    indices = motif_to_indices[motif_type]
    for G in graph_ref_list:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_ref.append(match_cnt / G.number_of_nodes())

        #hist, _ = np.histogram(
        #        motif_counts, bins=bins, density=False)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_ref.append(motif_temp)

    for G in graph_pred_list_remove_empty:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_pred.append(match_cnt / G.number_of_nodes())

        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)

    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian,
            is_hist=False)
    #print('-------------------------')
    #print(np.sum(total_counts_ref) / len(total_counts_ref))
    #print('...')
    #print(np.sum(total_counts_pred) / len(total_counts_pred))
    #print('-------------------------')
    return mmd_dist

def orbit_stats_all(mmdTrainTest, counts_Train, counts_Test, graph_pred_list):
    if mmdTrainTest != -1:
        total_counts_ref = []
        total_counts_ref2 = []
        total_counts_pred = []
    
        graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

        

        for G in graph_pred_list:
            try:
                orbit_counts = orca(G)
            except:
                continue
            orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
            total_counts_pred.append(orbit_counts_graph)

        total_counts_ref = counts_Train
        total_counts_ref2 = counts_Test
        total_counts_pred = np.array(total_counts_pred)
        mmd_dist_ref_pred = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian,
                is_hist=False, sigma=30.0)
        mmd_dist_ref2_pred = mmd.compute_mmd(total_counts_ref2, total_counts_pred, kernel=mmd.gaussian,
                is_hist=False, sigma=30.0)
        mmd_dist_ref_ref2 = mmdTrainTest

        print('-------------------------')
        print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
        print('...')
        print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
        print('-------------------------')
        return [mmd_dist_ref_pred, mmd_dist_ref2_pred, mmd_dist_ref_ref2, total_counts_ref,total_counts_ref2, total_counts_pred]
    else:
        return -1


def orbit_stats_allTrainTest(graph_ref_list, graph_ref2_list):
    total_counts_ref = []
    total_counts_ref2 = []
 

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_ref2_list:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref2.append(orbit_counts_graph)

    

    total_counts_ref = np.array(total_counts_ref)
    total_counts_ref2 = np.array(total_counts_ref2)
  
    mmd_dist_ref_ref2 = mmd.compute_mmd(total_counts_ref, total_counts_ref2, kernel=mmd.gaussian,
            is_hist=False, sigma=30.0)

    return [mmd_dist_ref_ref2, total_counts_ref,total_counts_ref2]






# --- 1. Define the GED calculation function 
# This function will be executed by each worker process.
# It should be self-contained and not rely on global variables for mutable state.
def calculate_single_ged(g1, g2): # No 'ged_cost_settings' here!
    """
    Computes GED between two graphs g1 and g2, focusing ONLY on the existence of edges.
    Node operations are set to have zero cost.
    Edge insertion/deletion cost is 1. Edge substitution cost is 0.
    """
    try:
        def always_match_nodes(n1, n2): return True
        def zero_node_subst_cost(n1, n2): return 0
        def node_del_cost_func(n): return 1
        def node_ins_cost_func(n): return 1
        edge_match_func = None
        edge_subst_cost_func = None
        def edge_del_cost_func(e): return 1
        def edge_ins_cost_func(e): return 1
        # No 'ged_cost_settings' used inside here either, as they are hardcoded.
        # Ensure the call to nx.graph_edit_distance uses the internal hardcoded values.
        cost = nx.graph_edit_distance(g1, g2,
                                    node_match=always_match_nodes,
                                    node_subst_cost=zero_node_subst_cost,
                                    node_del_cost=node_del_cost_func,
                                    node_ins_cost=node_ins_cost_func,
                                    edge_match=edge_match_func,
                                    edge_subst_cost=edge_subst_cost_func,
                                    edge_del_cost=edge_del_cost_func,
                                    edge_ins_cost=edge_ins_cost_func)
        return cost
    except Exception as e:
        print(f"Error computing GED between graphs: {e}")
        return float('inf')
# --- 2. Define the main worker function for each predicted graph ---
# --- 2. Define the main worker function for each predicted graph ---
# This function will be called for each predicted graph.
# It needs access to the *entire* list of reference graphs.
def GED_calculator_worker(predicted_graph, all_reference_graphs, ged_cost_settings_for_worker): # This signature is fine, it receives it
    """
    Computes the minimum GED for a single predicted graph against all reference graphs.
    """
    min_ged_for_current_predicted = float('inf')

    for ref_graph in all_reference_graphs:
        # --- FIX IS HERE: Remove the ged_cost_settings_for_worker argument ---
        current_ged = calculate_single_ged(predicted_graph, ref_graph) # Corrected call
        if current_ged < min_ged_for_current_predicted:
            min_ged_for_current_predicted = current_ged

    return min_ged_for_current_predicted

# --- NEW: Helper function to pack arguments for executor.map ---
def _map_worker_wrapper(args_tuple):
    """
    A wrapper function to unpack arguments for GED_calculator_worker when using executor.map.
    """
    predicted_graph, reference_graphs_list, ged_cost_settings_for_worker = args_tuple
    return GED_calculator_worker(predicted_graph, reference_graphs_list, ged_cost_settings_for_worker)



# --- 3. Main execution logic ---
def calculate_overall_ged_metrics(predicted_graphs, reference_graphs, ged_cost_settings=None, is_parallel=True):
    """
    Calculates overall GED metrics (min, max, mean, std) for predicted graphs
    against a set of reference graphs.

    Args:
        predicted_graphs (list): A list of networkx.Graph objects (predicted).
        reference_graphs (list): A list of networkx.Graph objects (reference).
        ged_cost_settings (dict, optional): Dictionary of custom GED cost functions.
        is_parallel (bool): Whether to use parallel processing.

    Returns:
        dict: A dictionary containing 'min_ged', 'max_ged', 'mean_ged', 'std_ged'.
    """
    if not predicted_graphs or not reference_graphs:
        raise ValueError("Predicted and reference graph lists cannot be empty.")

    min_geds_for_all_predicteds = []
    
    if is_parallel:
        # Using ProcessPoolExecutor for CPU-bound tasks like GED
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Prepare arguments as a list of tuples
            # Each tuple will be passed as a single argument to _map_worker_wrapper
            args_for_map = [(pg, reference_graphs, ged_cost_settings) for pg in predicted_graphs]

            # Use executor.map with the wrapper
            results_iterator = executor.map(_map_worker_wrapper, args_for_map)

            for min_ged_val in results_iterator:
                min_geds_for_all_predicteds.append(min_ged_val)
        

    else:
        print("Starting sequential GED computation...")
        for pg in predicted_graphs:
            min_ged_val = GED_calculator_worker(pg, reference_graphs, ged_cost_settings)
            min_geds_for_all_predicteds.append(min_ged_val)


    # Convert to numpy array for easy statistics
    min_geds_array = np.array(min_geds_for_all_predicteds)

    # Calculate statistics
    final_min_ged = np.min(min_geds_array)
    final_max_ged = np.max(min_geds_array)
    final_mean_ged = np.mean(min_geds_array)
    final_std_ged = np.std(min_geds_array)
    final_median_ged = np.median(min_geds_array)

    return [final_min_ged,final_max_ged,final_mean_ged,final_std_ged,final_median_ged]

# --- Example Usage ---
def GED_stats_all(graph_ref_list, graph_pred_list, graph_test_list, reference_results, is_parallel=True):
    results_tr = calculate_overall_ged_metrics(
        graph_pred_list,
        graph_ref_list,
        is_parallel=is_parallel
    )

    results_tst = calculate_overall_ged_metrics(
        graph_pred_list,
        graph_test_list,
        is_parallel=is_parallel
    )


    return [results, results_tst, reference_results]



def GED_stats_allTrainTest(graph_train_list, graph_test_list, is_parallel=True):
    results = calculate_overall_ged_metrics(
        graph_train_list,
        graph_test_list,
        is_parallel=is_parallel
    )
    return results
