from datetime import datetime
import os
import argparse

if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = f"tensorboard/run_{current_time}"

    #configure the args file if that is told from the terminal
    parser = argparse.ArgumentParser(description="Description of your program")
    
    # Define the expected arguments
     # Define the expected arguments
    parser.add_argument('--datasets', type=str, help='comma separated datasets: dataset1,dataset2,dataset3')
    parser.add_argument('--onlyEvaluate', action='store_true', help='True if you only want to execute the evaluation')
    parser.add_argument('--maximumEpochs', type=int, help='a number, which indicates which will be the maximum number of epochs')




    

    terminal = parser.parse_args()
    if terminal.datasets:
        print("datasets", terminal.datasets.split(',') )
        
 
    print("evaluate ",terminal.onlyEvaluate)




    
    if terminal.maximumEpochs is not None:
        print("epochs", terminal.maximumEpochs)
