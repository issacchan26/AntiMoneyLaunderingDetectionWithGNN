# AntiMoneyLaunderingDetectionWithGNN
This repo provides model training of Graph Attention Network in Anti Money Laundering Detection problem.  
Dataset: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

## Getting Started
Main dependencies are NumPy, PyTorch, PyG and pandas
Use the pip to install dependencies, you may use conda instead  
For PyG installation, you may use below code to install 
```bash
pip install torch_geometric
```

## Usage
Please create the corresponding folder before you run the script. 
Put the .csv file into raw folder, [dataset.py](dataset.py) will create "processed" folder with processed data once you run the [train.py](train.py).  
Make sure the directories are created as below:

```bash
├── data
│   ├── raw
├── dataset.py
├── model.py
└── train.py
```

## Data Preprocessing
All data preprocessing are done in [dataset.py](dataset.py), it used torch_geometric.data.InMemoryDataset as the dataset framework.  
Please note that [dataset.py](dataset.py) currently support single file processing.

## Model Training
Please change the path in line 8 to your local path, e.g. '/path/to/AntiMoneyLaunderingDetectionWithGNN/data'  
The hyperparameters used in [train.py](train.py):  
epoch = 100  
train_batch_size = 128
learning_rate=0.001  
optimizer: Adam

## Model Selection
This repo is using Graph Attention Network as backbone model, the model can be changed in [model.py](model.py)
