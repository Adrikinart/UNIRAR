import torch
import time

# Check available device
if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")

print("DEFAULT_DEVICE:", DEFAULT_DEVICE)

# Create a random tensor of shape (16, 2072) on the selected device
tensor = torch.rand(96, 20072, device=DEFAULT_DEVICE)

# Number of bins
start = time.time()

num_bins = 6

# Compute bin edges
min_val, max_val = tensor.min(), tensor.max()
bin_edges = torch.linspace(min_val, max_val, steps=num_bins + 1, device=DEFAULT_DEVICE)

# Assign each value to a bin
bin_indices = torch.bucketize(tensor, bin_edges, right=True) - 1

# Ensure indices are within valid range
bin_indices = bin_indices.clamp(0, num_bins - 1)

# One-hot encode and sum along last axis
histograms = torch.zeros((tensor.size(0), num_bins), device=DEFAULT_DEVICE)
histograms.scatter_add_(1, bin_indices, torch.ones_like(bin_indices, dtype=torch.float, device=DEFAULT_DEVICE))

print(f"time: {time.time() - start}")



start = time.time()
# Use same range for torch.histc
histograms2 = torch.stack(
    [torch.histc(tensor[i], bins=num_bins, min=min_val.item(), max=max_val.item()).to(DEFAULT_DEVICE)
     for i in range(tensor.shape[0])]
)
print(f"time: {time.time() - start}")

# Verify shapes
print("Shape of histograms:", histograms.shape)  # (16, 6)
print("Shape of histograms2:", histograms2.shape)  # (16, 6)

# Print both histograms
print("Histograms (scatter_add_):", histograms)
print("Histograms (histc):", histograms2)
