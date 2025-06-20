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



    

    terminal = parser.parse_args()
    if terminal.datasets:
        args.datasets = terminal.datasets.split(',') 
 
    args.evaluate = terminal.executeEvaluate
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
    
    print("datasets: ", terminal.datasets, ", execute evaluate: ", args.evaluate, ", models to train: ", args.notes, ", seed:",  terminal.seed)

