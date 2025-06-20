# This .py program is used to test a very big dataset which may be the ground truth dataset.



import argparse
import numpy as np
import os
import re
from random import shuffle

import eval.bilaketaHeuristikoak
import eval.stats
import utils
# import main.Args
from baselines.baseline_simple import *
import args
import create_graphs

class Args_evaluate():
    def __init__(self):
        args = Args()
        # loop over the settings
        # self.model_name_all = ['GraphRNN_MLP','GraphRNN_RNN','Internal','Noise']
        # self.model_name_all = ['E-R', 'B-A']
        self.model_name_all = args.notes
        # self.model_name_all = ['Baseline_DGMG']

        # list of dataset to evaluate
        # use a list of 1 element to evaluate a single dataset
        self.dataset_name_all =args.datasets
        # self.dataset_name_all = ['citeseer_small','caveman_small']
        # self.dataset_name_all = ['barabasi_noise0','barabasi_noise2','barabasi_noise4','barabasi_noise6','barabasi_noise8','barabasi_noise10']
        # self.dataset_name_all = ['caveman_small', 'ladder_small', 'grid_small', 'ladder_small', 'enzymes_small', 'barabasi_small','citeseer_small']
        
        self.epoch_start=args.epochs_test_start
        self.epoch_end=args.epochs
        self.epoch_step=args.epoch_test_step
        self.simulatedAnnealing_args =args.simulatedAnnealing_args
        self.EDA_args = args.EDA_args


def process_kron(kron_dir):
    txt_files = []
    for f in os.listdir(kron_dir):
        filename = os.fsdecode(f)
        if filename.endswith('.txt'):
            txt_files.append(filename)
        elif filename.endswith('.dat'):
            return utils.load_graph_list(os.path.join(kron_dir, filename))
    G_list = []
    for filename in txt_files:
        G_list.append(utils.snap_txt_output_to_nx(os.path.join(kron_dir, filename)))

    return G_list



def load_ground_truth(dir_input, dataset_name, model_name='GraphRNN_RNN'):
    ''' Read ground truth graphs.
    '''
    if not 'small' in dataset_name:
        hidden = 128
    else:
        hidden = 64
    if model_name=='Internal' or model_name=='Noise' or model_name=='B-A' or model_name=='E-R':
        fname_test = dir_input + 'GraphRNN_MLP' + '_' + dataset_name + '_' + str(args.num_layers) + '_' + str(
                hidden) + '_test_' + str(0) + '.dat'
    else:
        fname_test = dir_input + model_name + '_' + dataset_name + '_' + str(args.num_layers) + '_' + str(
                hidden) + '_test_' + str(0) + '.dat'
    try:
        graph_test = utils.load_graph_list(fname_test,is_real=True)
    except Exception as e:
    # This will catch any exception that inherits from Exception
        print(f"An error occurred: {e}")
        print("graphRNN3")
        print('Not found: ' + fname_test)
        logging.warning('Not found: ' + fname_test)
        return None
    return graph_test


def eval_single_list(graphs, dir_input, dataset_name):
    ''' Evaluate a list of graphs by comparing with graphs in directory dir_input.
    Args:
        dir_input: directory where ground truth graph list is stored
        dataset_name: name of the dataset (ground truth)
    '''
    graph_test = load_ground_truth(dir_input, dataset_name)
    graph_test_len = len(graph_test)
    graph_test = graph_test[int(0.8 * graph_test_len):] # test on a hold out test set
    mmd_degree = eval.stats.degree_stats(graph_test, graphs)
    mmd_clustering = eval.stats.clustering_stats(graph_test, graphs)
    try:
        mmd_4orbits = eval.stats.orbit_stats_all(graph_test, graphs)
    except:
        mmd_4orbits = -1
    print('deg: ', mmd_degree)
    print('clustering: ', mmd_clustering)
    print('orbits: ', mmd_4orbits)


def evaluation(args_evaluate,dir_input, dir_output, model_name_all, dataset_name_all, args, overwrite = True):
    ''' Evaluate the performance of a set of models on a set of datasets.
    '''
    fname_output = dir_output+"ground_true2"+'.csv'
    evaluation_epoch(dir_input,fname_output,args,is_clean=True, epoch_start=args_evaluate.epoch_start,epoch_end=args_evaluate.epoch_end,epoch_step=args_evaluate.epoch_step)



def evaluation_epoch(dir_input,fname_output,args, is_clean=True, epoch_start=1000,epoch_end=3001,epoch_step=100):
    with open(fname_output, 'w+') as f:
        f.write('model_dataset;sample_time;epoch;degree_train;clustering_train;orbits4_train;degree_test;clustering_test;orbits4_test;MIS_helburu_fn;MIS_helburu_fn_train\n')

        # TODO: Maybe refactor into a separate file/function that specifies THE naming convention
        # across main and evaluate

        hidden = 64
        # read real graph
        fname_test = args.datasets[0]
        try:
            graph_test = create_graphs.create_datasetName(args.datasets[0], args)
            graph_test2 = create_graphs.create_datasetName(args.datasets[0]+"_test", args)
        except Exception as e:
    # This will catch any exception that inherits from Exception
            print(f"An error occurred: {e}")
            print("graphRNN4")
            print('Not found: ' + fname_test)
            logging.warning('Not found: ' + fname_test)
            return None
        

        # evaluate MMD test
        mmd_degree = eval.stats.degree_stats(graph_test, graph_test2)
        mmd_clustering = eval.stats.clustering_stats(graph_test, graph_test2)
        try:
            mmd_4orbits = eval.stats.orbit_stats_all(graph_test, graph_test2)
        except:
            mmd_4orbits = -1
        MIS_heuristic_search = eval.stats.MIS_value_stats2(graph_test, graph_test2)
        # evaluate MMD train
        mmd_degree_train = mmd_degree
        mmd_clustering_train =  mmd_clustering
        try:
            mmd_4orbits_train =mmd_4orbits
        except:
            mmd_4orbits_train = -1
        MIS_heuristic_search_train = MIS_heuristic_search
        # write results
        f.write(str("ground true")+';'+
                str("0")+';'+
                str("0")+';'+
                str(mmd_degree_train)+';'+
                str(mmd_clustering_train)+';'+
                str(mmd_4orbits_train)+';'+ 
                str(mmd_degree)+';'+
                str(mmd_clustering)+';'+
                str(mmd_4orbits)+';'+
                str(MIS_heuristic_search)+';'+
                str(MIS_heuristic_search_train)+'\n')
        print('model_dataset',"ground true", 'degree',mmd_degree,'clustering',mmd_clustering,'orbits',mmd_4orbits, "MIS_function", MIS_heuristic_search, "MIS_function_train", MIS_heuristic_search_train)
        print("")


def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    args = Args()
    args_evaluate = Args_evaluate()

    parser = argparse.ArgumentParser(description='Evaluation arguments.')
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--export-real', dest='export', action='store_true')
    feature_parser.add_argument('--no-export-real', dest='export', action='store_false')
    feature_parser.add_argument('--kron-dir', dest='kron_dir', 
            help='Directory where graphs generated by kronecker method is stored.')

    parser.add_argument('--testfile', dest='test_file',
            help='The file that stores list of graphs to be evaluated. Only used when 1 list of '
                 'graphs is to be evaluated.')
    parser.add_argument('--dir-prefix', dest='dir_prefix',
            help='The file that stores list of graphs to be evaluated. Can be used when evaluating multiple'
                 'models on multiple datasets.')
    parser.add_argument('--graph-type', dest='graph_type',
            help='Type of graphs / dataset.')
    
    parser.set_defaults(export=False, kron_dir='', test_file='',
                        dir_prefix='',
                        graph_type=args.graph_type)
    prog_args = parser.parse_args()

    # dir_prefix = prog_args.dir_prefix
    # dir_prefix = "/dfs/scratch0/jiaxuany0/"
    dir_prefix = args.dir_input

    time_now = time.strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    #time_now = strftime("%Y-%m-%d %H:%M:%S", gmtime())   
    if not os.path.isdir('logs/'):
        os.makedirs('logs/')
    logging.basicConfig(filename='logs/evaluate' + time_now + '.log', level=logging.INFO)

    if prog_args.export:
        if not os.path.isdir('eval_results'):
            os.makedirs('eval_results')
        if not os.path.isdir('eval_results/ground_truth'):
            os.makedirs('eval_results/ground_truth')
        out_dir = os.path.join('eval_results/ground_truth', prog_args.graph_type)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        output_prefix = os.path.join(out_dir, prog_args.graph_type)
        print('Export ground truth to prefix: ', output_prefix)

        if prog_args.graph_type == 'grid':
            graphs = []
            for i in range(10,20):
                for j in range(10,20):
                    graphs.append(nx.grid_2d_graph(i,j))
            utils.export_graphs_to_txt(graphs, output_prefix)
        elif prog_args.graph_type == 'caveman':
            graphs = []
            for i in range(2, 3):
                for j in range(30, 81):
                    for k in range(10):
                        graphs.append(caveman_special(i,j, p_edge=0.3))
            utils.export_graphs_to_txt(graphs, output_prefix)
        elif prog_args.graph_type == 'citeseer':
            graphs = utils.citeseer_ego()
            utils.export_graphs_to_txt(graphs, output_prefix)
        else:
            # load from directory
            input_path = dir_prefix + real_graph_filename
            g_list = utils.load_graph_list(input_path)
            utils.export_graphs_to_txt(g_list, output_prefix)
    elif not prog_args.kron_dir == '':
        kron_g_list = process_kron(prog_args.kron_dir)
        fname = os.path.join(prog_args.kron_dir, prog_args.graph_type + '.dat')
        print([g.number_of_nodes() for g in kron_g_list])
        utils.save_graph_list(kron_g_list, fname)
    elif not prog_args.test_file == '':
        # evaluate single .dat file containing list of test graphs (networkx format)
        graphs = utils.load_graph_list(prog_args.test_file)
        eval_single_list(graphs, dir_input=dir_prefix+'graphs/', dataset_name='grid')
    ## if you don't try kronecker, only the following part is needed
    else:
        if not os.path.isdir(dir_prefix+'eval_results'):
            os.makedirs(dir_prefix+'eval_results')
        evaluation(args_evaluate,dir_input=dir_prefix+"graphs/", dir_output=dir_prefix+"eval_results/",
                   model_name_all=args_evaluate.model_name_all,dataset_name_all=args_evaluate.dataset_name_all,args=args,overwrite=True)



if __name__ == '__main__':
    main()