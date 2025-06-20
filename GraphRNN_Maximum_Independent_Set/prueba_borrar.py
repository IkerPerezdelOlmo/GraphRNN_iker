import os
import argparse

if __name__ == '__main__':
    


    #configure the args file if that is told from the terminal
    parser = argparse.ArgumentParser(description="Description of your program")
    

    parser.add_argument('--datasets', type=str, help='comma separated datasets: dataset1,dataset2,dataset3')
    parser.add_argument('--learningrate', type=float, help='a number, which indicates the learning rate for the training')
    parser.add_argument('--emaitzenCsvAnGehitu', type=str, help='a string, It will be added to the file name of the results.')
    parser.add_argument('--seed', type=int, help='seed for the training')
    parser.add_argument('--modelSize', type=int, help='This is the times the model will be enhanced')



    datasets = "none"
    terminal = parser.parse_args()
    if terminal.datasets:
        datasets = terminal.datasets.split(',') 
   
    lr = "none"
    if terminal.learningrate is not None:
        lr = str(terminal.learningrate)
        
    emaitzenCsvAnGehitu = "none"
    if terminal.emaitzenCsvAnGehitu is not None:
        emaitzenCsvAnGehitu = terminal.emaitzenCsvAnGehitu
    
    sd = "none"

    if terminal.seed is not None:
        sd = str(terminal.seed)


    ms = "none"

    if terminal.modelSize is not None:
        ms = str(terminal.modelSize)

   



    print(datasets, lr, emaitzenCsvAnGehitu, sd, ms)