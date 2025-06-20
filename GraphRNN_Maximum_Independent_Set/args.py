from eval.bilaketaHeuristikoak import funtzioEraikitzailea3, funtzioEraikitzailea2
### program configuration
class Args():
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Args, cls).__new__(cls)
            #to quit the error
            cls.tf_enable_onednn_opts = '0'

            ### if clean tensorboard
            cls.clean_tensorboard = False
            ### Which CUDA GPU device is used for training
            cls.cuda = 1
            ### Which GraphRNN model variant is used.
            # The simple version of Graph RNN
            # cls.note = 'GraphRNN_MLP'
            # The dependent Bernoulli sequence version of GraphRNN
            cls.note = 'GraphRNN_RNN'
            cls.notes = ['GraphRNN_RNN']

            ## for comparison, removing the BFS compoenent
            # cls.note = 'GraphRNN_MLP_nobfs'
            # cls.note = 'GraphRNN_RNN_nobfs'

            ### Which dataset is used to train the model
            # cls.graph_type = 'DD'
            # cls.graph_type = 'caveman'
            # cls.graph_type = 'caveman_small'
            # cls.graph_type = 'caveman_small_single'
            # cls.graph_type = 'community4'
            #cls.graph_type = 'grid'
            # cls.graph_type = 'grid_small'
            # cls.graph_type = 'ladder_small'

            # cls.graph_type = 'enzymes'
            # cls.graph_type = 'enzymes_small'
            # cls.graph_type = 'barabasi'
            # cls.graph_type = 'barabasi_small'
            # cls.graph_type = 'citeseer'
            # cls.graph_type = 'citeseer_small'
            cls.graph_type = 'random_erdos-renyi_50_M9_P3'
            
            # THE DATATSETS WE WILL TEST
            #cls.datasets = ['random_MIS_50_M9_P3', 'random_MIS_50_M9_P6', 'random_MIS_250_M9_P3', 'random_MIS_250_M9_P6', 'random_MIS_1000_M9_P3', 'random_MIS_1000_M9_P6', 'random_MIS_50_M20_P3', 'random_MIS_50_M20_P6', 'random_MIS_250_M20_P3', 'random_MIS_250_M20_P6', 'random_MIS_1000_M20_P3', 'random_MIS_1000_M20_P6']
            cls.datasets = ['random_erdos-renyi_50_M9_P3']
            cls.train_test = "" # si se quiere que sea train "", si no, poner "test"
            cls.evaluate = True # True if you want to make the evaluation in the main_iker program
            cls.onlyEvaluate = False
            cls.emaitzen_csv_an_gehitu = ""
            
            #cls.graph_type = 'random50'

            # cls.graph_type = 'barabasi_noise'
            # cls.noise = 10
            #
            # if cls.graph_type == 'barabasi_noise':
            #     cls.graph_type = cls.graph_type+str(cls.noise)

            # if none, then auto calculate
            cls.max_num_node = None # max number of nodes in a graph
            cls.max_prev_node = None # max previous node that looks back

            ### network config
            ## GraphRNN
            if 'small' in cls.graph_type:
                cls.parameter_shrink = 2
            else:
                cls.parameter_shrink = 1
            cls.hidden_size_rnn = int(128/cls.parameter_shrink) # hidden size for main RNN
            cls.hidden_size_rnn_output = 16 # hidden size for output RNN
            cls.embedding_size_rnn = int(64/cls.parameter_shrink) # the size for LSTM input
            cls.embedding_size_rnn_output = 8 # the embedding size for output rnn
            cls.embedding_size_output = int(64/cls.parameter_shrink) # the embedding size for output (VAE/MLP)

            cls.batch_size = 32 # normal: 32, and the rest should be changed accordingly
            cls.test_batch_size = 32
            cls.test_total_size = 50
            cls.num_layers = 4

            ### training config
            '''
            cls.num_workers = 4 # num workers to load data, default 4
            cls.batch_ratio = 32 # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
            cls.epochs = 3000 # now one epoch means cls.batch_ratio x batch_size
            cls.epochs_test_start = 100
            cls.epochs_test = 100
            cls.epochs_log = 100
            cls.epochs_save = 100
            
            '''
            cls.num_workers = 4 # num workers to load data, default 4
            cls.batch_ratio = 32 # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
            # EPOCHS BEFORE
            cls.epochs = 14000 # now one epoch means cls.batch_ratio x batch_size
            cls.epochsMin = 3000
            cls.epochs_test_start = 0
            cls.epochs_test = 10
            cls.epochs_log = 1
            cls.epochs_save = 1000
            cls.epoch_test_step = 1000
            cls.zeroEpoch = 0
            cls.getValidationLoss = False
            cls.epochsToTest = []
            #cls.epochs = 20 # now one epoch means cls.batch_ratio x batch_size
            #cls.epochs_test_start = 10
            #cls.epochs_test = 5
            #cls.epochs_log = 5
            #cls.epochs_save = 5
            #cls.epoch_test_step = 5
            

            cls.lr = 0.003
            cls.milestones = [400, 1000]
            cls.lr_rate = 0.3

            cls.sample_time = 2 # sample time in each time step, when validating



            cls.seed = 123

            ### output config
            # cls.dir_input = "/dfs/scratch0/jiaxuany0/"
            cls.dir_input = "./"
            cls.model_save_path = cls.dir_input+'model_save/' # only for nll evaluation
            cls.graph_save_path = cls.dir_input+'graphs/'
            cls.figure_save_path = cls.dir_input+'figures/'
            cls.timing_save_path = cls.dir_input+'timing/'
            cls.figure_prediction_save_path = cls.dir_input+'figures_prediction/'
            cls.nll_save_path = cls.dir_input+'nll/'


            cls.load = False # if load model, default lr is very low
            cls.load_epoch = 3000
            cls.save = True


            ### baseline config
            # cls.generator_baseline = 'Gnp'
            cls.generator_baseline = 'BA'

            # cls.metric_baseline = 'general'
            # cls.metric_baseline = 'degree'
            cls.metric_baseline = 'clustering'


        

            ##### ARGUMENTS FOR THE HEURISTIC SEARCH
            cls.EDA_args= [40, 20, 600, 0.3] #[100, 30, 7000, 0.3] # populazioTam, aukeratuTam, zenbat instantzia berri sortuko diren, probabilitatea (no se usa)
            cls.simulatedAnnealing_args= [10*10**-309, -1, 0.75, 0.85, 3000, funtzioEraikitzailea3] #[10*10**-309, -1, 0.75, 0.85, 0.0001, funtzioEraikitzailea3]
            cls.funtzioEraikitzaile_args= [funtzioEraikitzailea2]
        
        return cls._instance
    
     ### filenames to save intemediate and final outputs
    def fname(cls, eredua, dataseta):
        return cls.notes[eredua] + '_' + cls.datasets[dataseta] + cls.emaitzen_csv_an_gehitu + '_' + str(cls.num_layers) + '_' + str(cls.hidden_size_rnn) + '_'
    
    def fname_pred(cls, eredua, dataseta):
        return cls.notes[eredua] + '_' + cls.datasets[dataseta] + cls.emaitzen_csv_an_gehitu + '_'+str(cls.num_layers)+'_'+ str(cls.hidden_size_rnn)+'_pred_'
    
    def fname_train(cls, eredua, dataseta):
        return cls.notes[eredua] + '_' + cls.datasets[dataseta] + cls.emaitzen_csv_an_gehitu +'_'+str(cls.num_layers)+'_'+ str(cls.hidden_size_rnn)+'_train_'
    
    def fname_test(cls, eredua, dataseta):
        return cls.notes[eredua] + '_' + cls.datasets[dataseta] + cls.emaitzen_csv_an_gehitu + '_' + str(cls.num_layers) + '_' + str(cls.hidden_size_rnn) + '_test_'
    
    def fname_baseline(cls, eredua, dataseta):
        return cls.graph_save_path +  cls.datasets[dataseta] + cls.emaitzen_csv_an_gehitu + cls.generator_baseline+'_'+cls.metric_baseline
       ### filenames to save intemediate and final outputs
        #cls.fname = cls.note + '_' + cls.graph_type + '_' + str(cls.num_layers) + '_' + str(cls.hidden_size_rnn) + '_'
        #cls.fname_pred = cls.note+'_'+cls.graph_type+'_'+str(cls.num_layers)+'_'+ str(cls.hidden_size_rnn)+'_pred_'
        #cls.fname_train = cls.note+'_'+cls.graph_type+'_'+str(cls.num_layers)+'_'+ str(cls.hidden_size_rnn)+'_train_'
        #cls.fname_test = cls.note + '_' + cls.graph_type + '_' + str(cls.num_layers) + '_' + str(cls.hidden_size_rnn) + '_test_'
        #cls.fname_baseline = cls.graph_save_path + cls.graph_type + cls.generator_baseline+'_'+cls.metric_baseline

