# Anti Money Laundering Detection with Graph Attention Network
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

## Data analysis and visualization
This [Jupyter Notebook](anti-money-laundering-detection-with-gnn.ipynb) explains the feature engineering implemented and short summary in this repo.  
It also provides the data visualization and preprocessing pipeline, as well as dataset design details.

## Data Preprocessing
All data preprocessing are done in [dataset.py](dataset.py), it used torch_geometric.data.InMemoryDataset as the dataset framework.  
Please note that [dataset.py](dataset.py) currently support single file processing.

## Model Training
Please change the path in line 8 to your local path, e.g. '/path/to/AntiMoneyLaunderingDetectionWithGNN/data'  
The hyperparameters used in [train.py](train.py):  
epoch = 100  
train_batch_size = 256  
test_batch_size = 256  
learning_rate=0.0001  
optimizer: SGD

## Model Selection
This repo is using Graph Attention Network as backbone model, the model can be changed in [model.py](model.py)

## Reference 
Some of the feature engineering of this repo are referenced to below papers, highly recommend to read:
1. [Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics. arXiv preprint arXiv:1908.02591.](https://arxiv.org/pdf/1908.02591.pdf)
2. [Johannessen, F., & Jullum, M. (2023). Finding Money Launderers Using Heterogeneous Graph Neural Networks. arXiv preprint arXiv:2307.13499.](https://arxiv.org/pdf/2307.13499.pdf)
