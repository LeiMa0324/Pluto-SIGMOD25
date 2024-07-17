# Structure
##bert_pytorch
includes the model, trainer, predictor, log dataset class
##main
includes the main python files: vocab, train and predict
##output
includes the datasets: BGL, HDFS, ThunderBird and Spirit with training and test.
for each dataset, we provide the warm-up-model for fast embedding learning.

##run
includes the yaml configuration files to run

##sample
includes the sampler: co-teaching, fine and pluto. ITLM is embedded in the bert_pytorch.trainer.sample_selector.py

# Run config
we provide the running of 6 methods on the 4 datasets as follows:
```bash
% run a method
% method in ['coteaching','fine','ITLM','logbert','logbert-','pluto']
% dataset in ['bgl','hdfs','tbird','spirit']
python vocab_main.py --config='../run/{method}/{dataset}.yaml' % generate vocab
python train_main.py --config='../run/{method}/{dataset}.yaml' % train model
