import pandas as pd
import args
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from scipy.stats import gaussian_kde, entropy
import numpy as np
import random
import string
import eval.mmd as mmd
from collections import Counter
import argparse

def init_args():
    parser = argparse.ArgumentParser(description="Description of your program")
    
    # Define the expected arguments
     # Define the expected arguments
    parser.add_argument('--datasets', type=str, help='comma separated datasets: dataset1,dataset2,dataset3')
    parser.add_argument('--executeEvaluate', action='store_true', help='True if when finishing the training you want to make the evaluation')
    parser.add_argument('--modelsToTrain', type=str, help='comma separated models: model1,model2,model3')
    parser.add_argument('--maximumEpochs', type=int, help='a number, which indicates which will be the maximum number of epochs')
    parser.add_argument('--minimumEpochs', type=int, help='a number, which indicates which will be the minimum number of epochs')
    parser.add_argument('--epochsTestStart', type=int, help='a number')
    parser.add_argument('--epochsTest', type=int, help='a number')
    parser.add_argument('--epochsSave', type=int, help='a number')
    parser.add_argument('--epochTestStep', type=int, help='a number')
    

    terminal = parser.parse_args()
    if terminal.datasets:
        args.datasets = terminal.datasets.split(',') 
 
    args.evaluate = terminal.executeEvaluate

    if terminal.modelsToTrain:
        args.notes = terminal.modelsToTrain.split(',') 
    
    if terminal.maximumEpochs is not None:
        args.epochs = terminal.maximumEpochs

    if terminal.minimumEpochs is not None:
        args.epochsMin = terminal.minimumEpochs

    if terminal.epochsTestStart is not None:
        args.epochs_test_start = terminal.epochsTestStart

    if terminal.epochsTest is not None:
        args.epochs_test = terminal.epochsTest

    if terminal.epochsSave is not None:
        args.epochs_save = terminal.epochsSave

    if terminal.epochTestStep is not None:
        args.epoch_test_step = terminal.epochTestStep

def generate_hex_colors(n):
  """
  Generates n distinct hexadecimal color codes (#in1212 format).

  Args:
      n: The number of colors to generate.

  Returns:
      A list of n strings, where each string represents a hexadecimal color code.
  """
  colors = set()
  while len(colors) < n:
    # Generate random hex digits
    hex_code = "#" + ''.join(random.choices(string.hexdigits, k=6))

    # Check if the color is unique and add it to the set
    if hex_code not in colors:
      colors.add(hex_code)

  return list(colors)


arguments = args.Args()




#csv_file = "ground_true2"+".csv"
# Read the CSV file into a pandas DataFrame
#df = pd.read_csv("eval_results/"+csv_file, sep=';')
#print(df.values)
#print(df.columns)

#model_dataset_values = df['model_dataset'].tolist()
#sample_time_values = df['sample_time'].tolist()
#epoch_values = df['epoch'].tolist()
#degree_train_values = df['degree_train'].tolist()
#clustering_train_values = df['clustering_train'].tolist()
#orbits4_train_values = df['orbits4_train'].tolist()
#degree_test_values = df['degree_test'].tolist()
#clustering_test_values = df['clustering_test'].tolist()
#orbits4_test_values = df['orbits4_test'].tolist()
#MIS_helburu_fn_values = df['MIS_helburu_fn'].apply(ast.literal_eval).tolist()
#MIS_helburu_fn_train_values = df['MIS_helburu_fn_train'].apply(ast.literal_eval).tolist()

#print("st",sample_time_values,"ev",epoch_values,degree_train_values,clustering_train_values ,orbits4_train_values,degree_test_values,clustering_test_values,orbits4_test_values,MIS_helburu_fn_values,MIS_helburu_fn_train_values)

#trainMethod1 = MIS_helburu_fn_values[0][0]
#trainMethod2 = MIS_helburu_fn_values[0][1]
#testMethod1 = MIS_helburu_fn_values[0][2]
#testMethod2 = MIS_helburu_fn_values[0][3]

# Extract x and y coordinates from data
#x1, y1 = zip(*trainMethod1)
#x2, y2 = zip(*trainMethod2)

init_args()

graphics =[]
for model in arguments.notes:
    for dataset in arguments.datasets:
        csv_file = model+"_"+dataset+".csv"

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv("eval_results/"+csv_file, sep=';')
        #print(df.values)

        
        #model_dataset_values = df['model_dataset'].tolist()
        sample_time_values = df['sample_time'].tolist()
        epoch_values = df['epoch'].tolist()
        degree_train_values = df['degree_train'].tolist()
        clustering_train_values = df['clustering_train'].tolist()
        orbits4_train_values = df['orbits4_train'].tolist()
        degree_test_values = df['degree_test'].tolist()
        clustering_test_values = df['clustering_test'].tolist()
        orbits4_test_values = df['orbits4_test'].tolist()
        MIS_helburu_fn_values = df['MIS_helburu_fn'].apply(ast.literal_eval).tolist()
        MIS_helburu_fn_train_values = df['MIS_helburu_fn_train'].apply(ast.literal_eval).tolist()

        #print("st",sample_time_values,"ev",epoch_values,degree_train_values,clustering_train_values ,orbits4_train_values,degree_test_values,clustering_test_values,orbits4_test_values,MIS_helburu_fn_values,MIS_helburu_fn_train_values)
        
        method = []
        for i in range(len(sample_time_values)):
                trainMethod1 = MIS_helburu_fn_values[i][0]
                trainMethod2 = MIS_helburu_fn_values[i][1]
                testMethod1 = MIS_helburu_fn_values[i][2]
                testMethod2 = MIS_helburu_fn_values[i][3]

                x1, _ = zip(*trainMethod1)
                x2, _ = zip(*trainMethod2)
                x3, _ = zip(*testMethod1)
                x4, _ = zip(*testMethod2)

                # Count the frequencies of each unique value
                freq1 = Counter(x1)
                freq2 = Counter(x2)
                freq3 = Counter(x3)
                freq4 = Counter(x4)

                # Normalize counts to get probabilities
                total1 = sum(freq1.values())
                total2 = sum(freq2.values())
                total3 = sum(freq3.values())
                total4 = sum(freq4.values())

                prob1 = {k: v / total1 for k, v in freq1.items()}
                prob2 = {k: v / total2 for k, v in freq2.items()}
                prob3 = {k: v / total3 for k, v in freq3.items()}
                prob4 = {k: v / total4 for k, v in freq4.items()}


                keys1 = sorted(prob1.keys())
                keys2 = sorted(prob2.keys())
                keys3 = sorted(prob3.keys())
                keys4 = sorted(prob4.keys())

                # Plotting the probability distributions
                plt.figure(figsize=(10, 6))

                plt.plot(keys1, [prob1[k] for k in keys1], label="reference SA", color='black')
                plt.fill_between(keys1, [prob1[k] for k in keys1], color='black', alpha=0.3)

                plt.plot(keys2, [prob2[k] for k in keys2], label="reference EDA", color='yellow')
                plt.fill_between(keys2, [prob2[k] for k in keys2], color='yellow', alpha=0.3)

                plt.plot(keys3, [prob3[k] for k in keys3], label="predicted SA", color='#006400')
                plt.fill_between(keys3, [prob3[k] for k in keys3], color='#006400', alpha=0.3)

                plt.plot(keys4, [prob4[k] for k in keys4], label="predicted EDA", color='orangered')
                plt.fill_between(keys4, [prob4[k] for k in keys4], color='orangered', alpha=0.3)

                # Adding labels and title
                plt.xlabel("Value")
                plt.ylabel("Probability")
                plt.title(f"Probability Distribution for MIS_helburu_fn_values (epoch{epoch_values[i]}")

                  # Show legend
                plt.legend()

                # Display the plot
                plt.show()



                mmdSA_test = mmd.compute_mmd(np.array(trainMethod1), np.array(testMethod1), kernel=mmd.gaussian_emd,
                               sigma=1.0/10)
                mmdEDA_test = mmd.compute_mmd(np.array(trainMethod2), np.array(testMethod2), kernel=mmd.gaussian_emd,
                               sigma=1.0/10)

                trainMethod1 = MIS_helburu_fn_train_values[i][0]
                trainMethod2 = MIS_helburu_fn_train_values[i][1]
                testMethod1 = MIS_helburu_fn_train_values[i][2]
                testMethod2 = MIS_helburu_fn_train_values[i][3]

                mmdSA_train= mmd.compute_mmd(np.array(trainMethod1), np.array(testMethod1), kernel=mmd.gaussian_emd,
                               sigma=1.0/10)
                mmdEDA_train= mmd.compute_mmd(np.array(trainMethod2), np.array(testMethod2), kernel=mmd.gaussian_emd,
                               sigma=1.0/10)




                method.append((mmdSA_test+ mmdEDA_test + mmdSA_train+ mmdEDA_train)/4)


                print()
        graphics.append(method)



        for i in range(len(sample_time_values)):

                x1 = degree_test_values[i][1]
                x2 = degree_test_values[i][2]
                x3 = clustering_test_values[i][1]
                x4 = clustering_test_values[i][2]
                x5 = orbits4_test_values[i][1]
                x6 = orbits4_test_values[i][2]

      

                # Count the frequencies of each unique value
                freq1 = Counter(x1)
                freq2 = Counter(x2)
                freq3 = Counter(x3)
                freq4 = Counter(x4)
                freq5 = Counter(x5)
                freq6 = Counter(x6)

                # Normalize counts to get probabilities
                total1 = sum(freq1.values())
                total2 = sum(freq2.values())
                total3 = sum(freq3.values())
                total4 = sum(freq4.values())
                total5 = sum(freq5.values())
                total6 = sum(freq6.values())

                prob1 = {k: v / total1 for k, v in freq1.items()}
                prob2 = {k: v / total2 for k, v in freq2.items()}
                prob3 = {k: v / total3 for k, v in freq3.items()}
                prob4 = {k: v / total4 for k, v in freq4.items()}
                prob5 = {k: v / total5 for k, v in freq5.items()}
                prob6 = {k: v / total6 for k, v in freq6.items()}


                keys1 = sorted(prob1.keys())
                keys2 = sorted(prob2.keys())
                keys3 = sorted(prob3.keys())
                keys4 = sorted(prob4.keys())
                keys5 = sorted(prob5.keys())
                keys6 = sorted(prob6.keys())

                # Plotting the probability distributions
                plt.figure(figsize=(10, 6))

                plt.plot(keys1, [prob1[k] for k in keys1], label="reference degree", color='black')
                plt.fill_between(keys1, [prob1[k] for k in keys1], color='black', alpha=0.3)

                plt.plot(keys2, [prob2[k] for k in keys2], label="reference degree", color='yellow')
                plt.fill_between(keys2, [prob2[k] for k in keys2], color='yellow', alpha=0.3)

                plt.plot(keys3, [prob3[k] for k in keys3], label="predicted clustering", color='#006400')
                plt.fill_between(keys3, [prob3[k] for k in keys3], color='#006400', alpha=0.3)

                plt.plot(keys4, [prob4[k] for k in keys4], label="predicted clustering", color='orangered')
                plt.fill_between(keys4, [prob4[k] for k in keys4], color='orangered', alpha=0.3)

                plt.plot(keys5, [prob5[k] for k in keys5], label="predicted orbit", color='orangered')
                plt.fill_between(keys4, [prob5[k] for k in keys5], color='orangered', alpha=0.3)

                plt.plot(keys6, [prob6[k] for k in keys6], label="predicted orbit", color='orangered')
                plt.fill_between(keys6, [prob6[k] for k in keys6], color='orangered', alpha=0.3)

                # Adding labels and title
                plt.xlabel("Value")
                plt.ylabel("Probability")
                plt.title(f"Probability Distribution for orbit degree and clustering (epoch{epoch_values[i]})")

                  # Show legend
                plt.legend()

                # Display the plot
                plt.show()



plt.figure(figsize=(10, 6))

colors = generate_hex_colors(len(graphics))



epochs = epoch_values
for i in range(len(graphics)):
    plt.plot(epochs, graphics[i], marker='o', linestyle='-', color=colors[i], label='Method: ''+arguments.notes[i%len(arguments.notes)]+ "dataset: "+ arguments.datasets[i%len(arguments.datasets)]')

# Add titles and labels
plt.title('KL Divergence Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('KL Divergence')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()



plt.figure(figsize=(10, 6))

colors = generate_hex_colors(len(graphics))




plt.plot(epochs, degree_train_values, marker='o', linestyle='-', color=colors[i], label='Method: degree train')
plt.plot(epochs, clustering_train_values, marker='o', linestyle='-', color=colors[i], label='Method: clustering tarin')
plt.plot(epochs, orbits4_train_values, marker='o', linestyle='-', color=colors[i], label='Method: ordbit train')
plt.plot(epochs, degree_test_values, marker='o', linestyle='-', color=colors[i], label='Method: dgree test')
plt.plot(epochs, clustering_test_values, marker='o', linestyle='-', color=colors[i], label='Method: clustering test')
plt.plot(epochs, orbits4_test_values, marker='o', linestyle='-', color=colors[i], label='Method: orbit test')

# Add titles and labels
plt.title('all metrics')
plt.xlabel('Epochs')
plt.ylabel('KL Divergence')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

print("clustering test", clustering_test_values, "clustering train", clustering_train_values, )
