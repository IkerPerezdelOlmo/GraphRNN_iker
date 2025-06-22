
Instance generation for MIS problem using GraphRNN
=======

## Installation
Install PyTorch following the instuctions on the [official website](https://pytorch.org/). The code has been tested over PyTorch 0.2.0 and 0.4.0 versions.
```bash
conda install pytorch torchvision cuda90 -c pytorch
```
Then install the other dependencies.
```bash
pip install -r cleaned_requirements.txt
```

## Test run
```bash
python main_iker.py
```

## Code description
For the GraphRNN model:
`main_iker.py` is the main executable file, and specific arguments are set in `args.py`.
`train.py` includes training iterations and calls `model.py` and `data.py`
`create_graphs.py` is where we prepare target graph datasets.



Parameter setting:
To adjust the hyper-parameter and input arguments to the model, modify the fields of `args.py`
accordingly.
For example, `args.cuda` controls which GPU is used to train the model, and `args.graph_type`
specifies which dataset is used to train the generative model. See the documentation in `args.py`
for more detailed descriptions of all fields.


## Evaluation
The evaluation is done in `evaluateIker.py`, where user can choose which settings to evaluate.
To evaluate how close the generated graphs are to the ground truth set, we use MMD (maximum mean discrepancy) to calculate the divergence between two _sets of distributions_ related to
the ground truth and generated graphs.
Three types of distributions are chosen: degree distribution, clustering coefficient distribution.
Both of which are implemented in `eval/stats.py`, using multiprocessing python
module. One can easily extend the evaluation to compute MMD for other distribution of graphs.

We also compute the orbit counts for each graph, represented as a high-dimensional data point. We then compute the MMD
between the two _sets of sampled points_ using ORCA (see http://www.biolab.si/supp/orca/orca.html) at `eval/orca`. 
One first needs to compile ORCA by 
```bash
g++ -O2 -std=c++11 -o orca orca.cpp` 
```
in directory `eval/orca`.
(the binary file already in repo works in Ubuntu). 

To evaluate, run 
```bash
python evaluate.py
```
Arguments specific to evaluation is specified in class
`evaluate.Args_evaluate`. Note that the field `Args_evaluate.dataset_name_all` must only contain
datasets that are already trained, by setting args.graph_type to each of the datasets and running
`python main_iker.py`.



## Misc
Jesse Bettencourt and Harris Chan have made a great [slide](https://duvenaud.github.io/learn-discrete/slides/graphrnn.pdf) introducing GraphRNN in Prof. David Duvenaud’s seminar course [Learning Discrete Latent Structure](https://duvenaud.github.io/learn-discrete/).

>>>>>>> 1ef475d (first commit)
