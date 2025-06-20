import argparse
import numpy as np
import os
import re
from random import shuffle

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
        self.emaitzen_csv_an_gehitu = args.emaitzen_csv_an_gehitu
        self.epochsToTest = args.epochsToTest
        self.model_size = args.num_layers/4


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
        hidden = 128 * int(args_evaluate.model_size)
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


def evaluation_epoch(args_evaluate, model_dataset, dir_input, fname_output, model_name, dataset_name, args, is_clean=True, epoch_start=1000,epoch_end=3001,epoch_step=100, epochsToTest = []):
    with open(fname_output, 'w+') as f:
        #f.write('model_dataset;sample_time;epoch;degree_test;clustering_test;orbits4_test;MIS_helburu_fn;GED\n')
        f.write('model_dataset;sample_time;epoch;degree_test;clustering_test;orbits4_test;MIS_helburu_fn\n')

        # TODO: Maybe refactor into a separate file/function that specifies THE naming convention
        # across main and evaluate
        if not 'small' in dataset_name:
            hidden = 128 * int(args_evaluate.model_size)
        else:
            hidden = 64
        
        
        graph_train =  create_graphs.create_datasetName(dataset_name, args)
        graph_test =  create_graphs.create_datasetName(dataset_name+"_test", args)
        
        graph_test_aver = 0
        for graph in graph_test:
            graph_test_aver+=graph.number_of_nodes()
        graph_test_aver /= len(graph_test)
        print('test average len',graph_test_aver)


        # get performance for proposed approaches
        if 'GraphRNN' in model_name:
            mmdTrainTestDegree, degreeTrain, degreeTest = eval.stats.degree_statsTrainTest(graph_train, graph_test)
            mmdTrainTestClustering, clusteringTrain, clsuteringTest = eval.stats.clustering_statsTrainTest(graph_train, graph_test)
            try:
                mmdTrainTestOrbit, counts_Train,counts_Test = eval.stats.orbit_stats_allTrainTest(graph_train, graph_test)
            except:
                mmdTrainTestOrbit, counts_Train,counts_Test = -1, -1, -1
            heuristicSearchTrain, heuristicSearchTest, TraintestMMDs  = eval.stats.MIS_value_statsTrainTest(graph_train, graph_test)

            #GED_statistical_results = eval.stats.GED_stats_allTrainTest(graph_train, graph_test)
            # read test graph

            if epochsToTest == []:
                epochs = [epoch for epoch in range(epoch_start,epoch_end,epoch_step)]
            else:
                epochs = epochsToTest
            for epoch in epochs:
                for sample_time in range(1,2):
                    # get filename
                    fname_pred = dir_input + model_name + '_' + dataset_name + args_evaluate.emaitzen_csv_an_gehitu  + '_' + str(args.num_layers) + '_' + str(hidden) + '_pred_' + str(epoch) + '_' + str(sample_time) + '.dat'
                    # load graphs
                    try:
                        graph_pred = utils.load_graph_list(fname_pred,is_real=False) # default False
                    except:
                        print("graphRNN ", args_evaluate.emaitzen_csv_an_gehitu)     
                        print('Not found: '+ fname_pred)
                        logging.warning('Not found: '+ fname_pred)
                        continue
                    # clean graphs
                    #if is_clean:
                    #    graph_test, graph_pred = clean_graphs(graph_test, graph_pred)
                    #else:
                    #    shuffle(graph_pred)
                    #    graph_pred = graph_pred[0:len(graph_test)]
                    print('len graph_test', len(graph_test))
                    print('len graph_train', len(graph_train))
                    print('len graph_pred', len(graph_pred))

                    graph_pred_aver = 0
                    for graph in graph_pred:
                        graph_pred_aver += graph.number_of_nodes()
                    graph_pred_aver /= len(graph_pred)
                    print('pred average len', graph_pred_aver)

                    # evaluate MMD test
                    mmd_degree = eval.stats.degree_stats(mmdTrainTestDegree, degreeTrain, degreeTest, graph_pred)
                    mmd_clustering = eval.stats.clustering_stats(mmdTrainTestClustering, clusteringTrain, clsuteringTest, graph_pred)
                    try:
                        mmd_4orbits = eval.stats.orbit_stats_all(mmdTrainTestOrbit, counts_Train, counts_Test, graph_pred)
                    except:
                        mmd_4orbits = -1
                    MIS_heuristic_search = eval.stats.MIS_value_stats(heuristicSearchTrain, heuristicSearchTest, TraintestMMDs, graph_pred)
                    # evaluate MMD train
                    #GED_statistical_results = eval.stats.GED_stats_all(graph_train, graph_pred, graph_test, GED_statistical_results)
                    
                    # write results
                    f.write(str(model_dataset)+';'+ \
                            str(sample_time)+';'+ \
                            str(epoch)+';'+ \
                            str(mmd_degree)+';'+ \
                            str(mmd_clustering)+';'+ \
                            str(mmd_4orbits)+';'+ \
                            str(MIS_heuristic_search)+ ';' + \
                            # str(GED_statistical_results)+ \
                            '\n')
                    # print('model_dataset',model_dataset, 'degree',mmd_degree,'clustering',mmd_clustering,'orbits',mmd_4orbits, "MIS_function", MIS_heuristic_search, 'GED', GED_statistical_results)
                    print('model_dataset',model_dataset, 'degree',mmd_degree,'clustering',mmd_clustering,'orbits',mmd_4orbits, "MIS_function", MIS_heuristic_search)
                    print("")


def evaluation(args_evaluate,dir_input, dir_output, model_name_all, dataset_name_all, args, overwrite = True):
    ''' Evaluate the performance of a set of models on a set of datasets.
    '''
    for model_name in model_name_all:
        for dataset_name in dataset_name_all:
            # check output exist
            fname_output = dir_output+model_name+'_'+dataset_name+ args_evaluate.emaitzen_csv_an_gehitu + '.csv'
            print('processing: '+dir_output + model_name + '_' + dataset_name + args_evaluate.emaitzen_csv_an_gehitu +'.csv')
            logging.info('processing: '+dir_output + model_name + '_' + dataset_name + args_evaluate.emaitzen_csv_an_gehitu +'.csv')
            if overwrite==False and os.path.isfile(fname_output):
                print(dir_output+model_name+'_'+dataset_name+ args_evaluate.emaitzen_csv_an_gehitu +'.csv exists!')
                logging.info(dir_output+model_name+'_'+dataset_name+ args_evaluate.emaitzen_csv_an_gehitu +'.csv exists!')
                continue
            evaluation_epoch(args_evaluate, model_name+'_'+dataset_name, dir_input,fname_output,model_name,dataset_name,args,is_clean=True, epoch_start=args_evaluate.epoch_start,epoch_end=args_evaluate.epoch_end,epoch_step=args_evaluate.epoch_step, epochsToTest = args_evaluate.epochsToTest)



def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    args = Args()
    args_evaluate = Args_evaluate()

    # dir_prefix = prog_args.dir_prefix
    # dir_prefix = "/dfs/scratch0/jiaxuany0/"
    dir_prefix = args.dir_input

    time_now = time.strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    #time_now = strftime("%Y-%m-%d %H:%M:%S", gmtime())   
    if not os.path.isdir('logs/'):
        os.makedirs('logs/')
    logging.basicConfig(filename='logs/evaluate' + time_now + '.log', level=logging.INFO)

    evaluation(args_evaluate,dir_input=dir_prefix+"graphs/", dir_output=dir_prefix+"eval_results/",
                   model_name_all=args_evaluate.model_name_all,dataset_name_all=args_evaluate.dataset_name_all,args=args,overwrite=True)



if __name__ == '__main__':
    main()
