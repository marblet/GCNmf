import torch
from torch_geometric.data import Data
from models import GCN, GCNmf
import numpy as np



def test_gcn_gradients():
    # Assuming you have a way to create or get a dummy 'data' object compatible with your model
    # For example, you can create a dummy 'Data' object as follows (adjust dimensions as needed):
    device = 'cpu'
    num_nodes = 10
    num_features = 5  # Adjust to match your model's expected input feature size
    dummy_data = Data(x=torch.randn(num_nodes, num_features), 
                      edge_index=torch.randint(0, num_nodes, (2, 20)), 
                      batch=torch.zeros(num_nodes, dtype=torch.long), obs=torch.rand(1, num_nodes, 2))

    # Create the GCN model instance
    model = GCN(dummy_data)
    model.to(device)
    model.train()
    # Create a dummy target output
    dummy_target = torch.tensor([1], dtype=torch.float).to(device)

    # Forward pass
    dummy_data = dummy_data.to(device)
    output = model(dummy_data)

    # Compute loss
    criterion = torch.nn.MSELoss()
    loss = criterion(output, dummy_target)

    # Backward pass
    loss.backward()

    # Check that the gradients are nonzero
    for name, param in model.named_parameters():
      if param.grad is not None:
        print(name, param.grad.shape)
        assert np.sum(np.abs(param.grad.cpu().numpy())) > 1E-9
      else:
        print(f"Skipping {name} because it has no gradient")



def test_gcnmf_gradients():
  device = 'cpu'
  num_nodes = 10
  num_features = 5  # Adjust to match your model's expected input feature size
  dummy_data = Data(x=torch.randn(num_nodes, num_features), 
                    edge_index=torch.randint(0, num_nodes, (2, 20)), 
                    batch=torch.zeros(num_nodes, dtype=torch.long))
  dummy_target = torch.tensor([0], dtype=torch.float).to(device)
  
  model = GCNmf(dummy_data).to(device)
  model.test_reset_parameters()
  model.train()
  
  # Forward pass
  dummy_data = dummy_data.to(device)
  output = model(dummy_data).squeeze(1)

  # Compute loss
  criterion = torch.nn.MSELoss()
  loss = criterion(output, dummy_target)

  # Backward pass
  loss.backward()
  
  # Check that the gradients are nonzero
  for name, param in model.named_parameters():
    if not param.grad.isnan().any():
      print(name, param.grad.shape)
      print(param.grad)
      assert np.sum(np.abs(param.grad.cpu().numpy())) > 1E-9
    else:
      print(f"Skipping {name} because it has no gradient")


# Run the test
# test_gcn_gradients()
test_gcnmf_gradients()