import torch
from model import GAT
from dataset import AMLtoGraph
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = AMLtoGraph('/path/to/AntiMoneyLaunderingDetectionWithGNN/data')
data = dataset[0]
epoch = 100

model = GAT(in_channels=data.num_features, hidden_channels=16, out_channels=1, heads=8)
model = model.to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

split = T.RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0)
data = split(data)

train_loader = NeighborLoader(
    data,
    num_neighbors=[30] * 2,
    batch_size=256,
    input_nodes=data.train_mask,
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[30] * 2,
    batch_size=256,
    input_nodes=data.val_mask,
)

for i in range(epoch):
    total_loss = 0
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        data.to(device)
        pred = model(data.x, data.edge_index, data.edge_attr)
        ground_truth = data.y
        loss = criterion(pred, ground_truth.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
    if epoch%10 == 0:
        print(f"Epoch: {i:03d}, Loss: {total_loss:.4f}")
        model.eval()
        acc = 0
        total = 0
        with torch.no_grad():
            for test_data in test_loader:
                test_data.to(device)
                pred = model(test_data.x, test_data.edge_index, test_data.edge_attr)
                ground_truth = test_data.y
                correct = (pred == ground_truth.unsqueeze(1)).sum().item()
                total += len(ground_truth)
                acc += correct
            acc = acc/total
            print('accuracy:', acc)

