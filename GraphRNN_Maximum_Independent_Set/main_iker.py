from train import *
from datetime import datetime
import torch
import os
import evaluateIker
import argparse
import create_graphs

if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    # All necessary arguments are defined in args.py
    args = Args()
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = str(args.tf_enable_onednn_opts)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    print('File name prefix',args.fname(0, 0))
    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")



    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = f"tensorboard/run_{current_time}"
    configure(logdir, flush_secs=5)

    #configure the args file if that is told from the terminal
    parser = argparse.ArgumentParser(description="Description of your program")
    
    # Define the expected arguments
     # Define the expected arguments
    parser.add_argument('--datasets', type=str, help='comma separated datasets: dataset1,dataset2,dataset3')
    parser.add_argument('--executeEvaluate', action='store_true', help='True if when finishing the training you want to make the evaluation')
    parser.add_argument('--onlyEvaluate', action='store_true', help='True if you only want to execute the evaluation')
    parser.add_argument('--modelsToTrain', type=str, help='comma separated models: model1,model2,model3')
    parser.add_argument('--maximumEpochs', type=int, help='a number, which indicates which will be the maximum number of epochs')
    parser.add_argument('--minimumEpochs', type=int, help='a number, which indicates which will be the minimum number of epochs if it not specified it will be the same as maximum epochs.')
    parser.add_argument('--epochsTestStart', type=int, help='a number')
    parser.add_argument('--epochsSave', type=int, help='a number')
    parser.add_argument('--epochTestStep', type=int, help='a number')
    parser.add_argument('--milestones', type=int, help='comma separated milestones: model1,model2,model3. This indicates when the learning rate decay will be applicated.')
    parser.add_argument('--learningrate', type=float, help='a number, which indicates the learning rate for the training')
    parser.add_argument('--learningrateDecay', type=float, help='a number, which indicates by which number will the learning rate be multiplied in the milestones')
    parser.add_argument('--emaitzenCsvAnGehitu', type=str, help='a string, It will be added to the file name of the results.')
    parser.add_argument('--getValidationLoss', action='store_true', help='True if we want to get validationLoss')
    parser.add_argument('--zeroEpoch', type=int, help='An early epoch where you want to evaluate predicted graphs')
    parser.add_argument('--epochsToTest', type=str, help='comma separated epochs. Those are the ones that are going to be tested')
    parser.add_argument('--seed', type=int, help='seed for the training')
    parser.add_argument('--modelSize', type=int, help='This is the times the model will be enhanced')




    

    terminal = parser.parse_args()
    if terminal.datasets:
        args.datasets = terminal.datasets.split(',') 
   
    print(args.datasets) 
    # args.evaluate = terminal.executeEvaluate
    args.onlyEvaluate = terminal.onlyEvaluate

    if terminal.modelsToTrain:
        args.notes = terminal.modelsToTrain.split(',') 

    if terminal.epochsToTest:
        args.epochsToTest = terminal.epochsToTest.split(",")
    
    if terminal.maximumEpochs is not None:
        args.epochs = terminal.maximumEpochs

    if terminal.milestones:
        milestones = [i for i in range(terminal.milestones, args.epochs, terminal.milestones)]
        args.milestones = milestones 

    if terminal.minimumEpochs is not None:
        args.epochsMin = terminal.minimumEpochs
    else: 
        args.epochsMin = args.epochs

    if terminal.epochsTestStart is not None:
        args.epochs_test_start = terminal.epochsTestStart

    if terminal.epochsSave is not None:
        args.epochs_save = terminal.epochsSave

    if terminal.epochTestStep is not None:
        args.epoch_test_step = terminal.epochTestStep
        args.epochs_test = terminal.epochTestStep

    if terminal.learningrate is not None:
        args.lr = terminal.learningrate
        
    if terminal.learningrateDecay is not None:
        args.lr_rate = terminal.learningrateDecay

    if terminal.emaitzenCsvAnGehitu is not None:
        args.emaitzen_csv_an_gehitu = terminal.emaitzenCsvAnGehitu
    
    args.getValidationLoss = terminal.getValidationLoss

    if terminal.zeroEpoch is not None:
        args.zeroEpoch = terminal.zeroEpoch

    if terminal.seed is not None:
        args.seed = terminal.seed

    # change model architecture
    if terminal.modelSize is not None:
        args.num_layers *= terminal.modelSize
        args.hidden_size_rnn *= terminal.modelSize
        args.hidden_size_rnn_output *= terminal.modelSize
        args.embedding_size_rnn *= terminal.modelSize
        args.embedding_size_rnn_output *= terminal.modelSize
        args.embedding_size_output *= terminal.modelSize

    print("Num layers= ", args.num_layers, ", hidden size = ", args.hidden_size_rnn)
    print("seed", args.seed)
    print("your datasets will be: ", args.datasets)
    print("Will the evaluation be executed? ",args.evaluate)
    print(f"Epochs: {args.epochs}, minimum epochs: {args.epochsMin}, epochs_test_start: {args.epochs_test_start}, epochs_test: {args.epochs_test}, epochs_save: {args.epochs_save}, epochs_test_step: {args.epoch_test_step}")
    print(args.onlyEvaluate)

    if  args.onlyEvaluate == False:

        for dataset_index in range(len(args.datasets)):
            #for notes in args.notes:
            for model_index in range(len(args.notes)):
                graphs = create_graphs.create_datasetName(args.datasets[dataset_index], args)
                print("LOADED")
                # split datasets
                random.seed(args.seed)
                shuffle(graphs)
                graphs_len = len(graphs)
                graphs_test = graphs[int(0.8 * graphs_len):]
                graphs_train = graphs[0:int(0.8*graphs_len)]
                graphs_validate = graphs[0:int(0.2*graphs_len)]
                # if use pre-saved graphs
                # dir_input = "/dfs/scratch0/jiaxuany0/graphs/"
                # fname_test = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
                #     args.hidden_size_rnn) + '_test_' + str(0) + '.dat'
                # graphs = load_graph_list(fname_test, is_real=True)
                # graphs_test = graphs[int(0.8 * graphs_len):]
                # graphs_train = graphs[0:int(0.8 * graphs_len)]
                # graphs_validate = graphs[int(0.2 * graphs_len):int(0.4 * graphs_len)]


                graph_validate_len = 0
                for graph in graphs_validate:
                    graph_validate_len += graph.number_of_nodes()
                graph_validate_len /= len(graphs_validate)
                print('graph_validate_len', graph_validate_len)

                graph_test_len = 0
                for graph in graphs_test:
                    graph_test_len += graph.number_of_nodes()
                graph_test_len /= len(graphs_test)
                print('graph_test_len', graph_test_len)


                args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
                max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
                min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

                # args.max_num_node = 2000
                # show graphs statistics
                print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
                print('max number node: {}'.format(args.max_num_node))
                print('max/min number edge: {}; {}'.format(max_num_edge,min_num_edge))
                print('max previous node: {}'.format(args.max_prev_node))

                # save ground truth graphs
                ## To get train and test set, after loading you need to manually slice
                save_graph_list(graphs, args.graph_save_path + args.fname_train(model_index, dataset_index) + '0.dat')
                save_graph_list(graphs, args.graph_save_path + args.fname_test(model_index, dataset_index) + '0.dat')
                print('train and test graphs saved at: ', args.graph_save_path + args.fname_test(model_index, dataset_index) + '0.dat')

                ### comment when normal training, for graph completion only
                # p = 0.5
                # for graph in graphs_train:
                #     for node in list(graph.nodes()):
                #         # print('node',node)
                #         if np.random.rand()>p:
                #             graph.remove_node(node)
                    # for edge in list(graph.edges()):
                    #     # print('edge',edge)
                    #     if np.random.rand()>p:
                    #         graph.remove_edge(edge[0],edge[1])


                ### dataset initialization
                if 'nobfs' in args.note:
                    print('nobfs')
                    dataset = Graph_sequence_sampler_pytorch_nobfs(graphs_train, max_num_node=args.max_num_node)
                    args.max_prev_node = args.max_num_node-1
                if 'barabasi_noise' in args.graph_type:
                    print('barabasi_noise')
                    dataset = Graph_sequence_sampler_pytorch_canonical(graphs_train,max_prev_node=args.max_prev_node)
                    args.max_prev_node = args.max_num_node - 1
                else:
                    dataset = Graph_sequence_sampler_pytorch(graphs_train,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
                    if args.getValidationLoss:
                        dataset_validate = Graph_sequence_sampler_pytorch(graphs_validate,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
                    
                sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                               num_samples=args.batch_size*args.batch_ratio, replacement=True)
                if args.getValidationLoss:
                    sample_strategy_validate = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset_validate) for i in range(len(dataset_validate))],
                                                                                num_samples=args.batch_size*args.batch_ratio, replacement=True)
                
                dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                    sampler=sample_strategy)
                if args.getValidationLoss:
                    dataset_loader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=args.batch_size, num_workers=args.num_workers,
                                                        sampler=sample_strategy_validate)
                else:
                    dataset_loader_validate = None

                ### model initialization
                ## Graph RNN VAE model
                # lstm = LSTM_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_lstm,
                #                   hidden_size=args.hidden_size, num_layers=args.num_layers).cuda()

                if 'GraphRNN_VAE_conditional' in args.notes[model_index]:
                    rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                                    hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                                    has_output=False).cuda()
                    output = MLP_VAE_conditional_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).cuda()
                elif 'GraphRNN_MLP' in args.notes[model_index]:
                    rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                                    hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                                    has_output=False).cuda()
                    output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).cuda()
                elif 'GraphRNN_RNN' in args.notes[model_index]:

                    print(f"PyTorch version: {torch.__version__}")
                    print(f"CUDA version: {torch.version.cuda}")
                    print(f"cuDNN version: {torch.backends.cudnn.version()}")
                    print(f"CUDA available: {torch.cuda.is_available()}")

                    rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                                    hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                                    has_output=True, output_size=args.hidden_size_rnn_output).cuda()
                    output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                                    hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                                    has_output=True, output_size=1).cuda()

                ### start training
                print("start training")
                train(args, dataset_loader, rnn, output, model_index, dataset_index, args.getValidationLoss, dataset_loader_validate)

                ### graph completion
                # train_graph_completion(args,dataset_loader,rnn,output)

                ### nll evaluation
                # train_nll(args, dataset_loader, dataset_loader, rnn, output, max_iter = 200, graph_validate_len=graph_validate_len,graph_test_len=graph_test_len)
    print("hi")
    print(args.evaluate)
    

    if args.evaluate:
        evaluateIker.main()

