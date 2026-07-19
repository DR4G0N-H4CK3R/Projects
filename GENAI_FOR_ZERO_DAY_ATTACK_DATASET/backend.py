from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import io
import json

app = FastAPI(title="WGAN-GP Cyber-Twin API", description="Dedicated PyTorch Training Backend")

# --- Global State for Demo Purposes ---
# In production, use Redis or a database to track jobs
training_state = {
    "status": "idle", # idle, training, completed, failed
    "progress": 0.0,
    "current_epoch": 0,
    "total_epochs": 0,
    "c_loss": 0.0,
    "g_loss": 0.0
}

# Store models globally to generate data later
models = {"netG": None, "netC": None, "features": None, "seq_len": 10, "latent_dim": 32}
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Architectures (Replicated for backend independence) ---
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1)**0.5)
        return torch.matmul(self.softmax(scores), v)

class TrafficGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = SelfAttention(hidden_dim * 2)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, output_dim), nn.Tanh())
        
    def forward(self, z):
        out, _ = self.lstm(z)
        out = self.attention(out)
        return self.fc(out)

class TrafficCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = SelfAttention(hidden_dim * 2)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.attention(out)
        return torch.mean(self.fc(out), dim=1)

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    batch_size, seq_len, num_features = real_samples.size()
    alpha = torch.rand(batch_size, 1, 1, device=device).expand(batch_size, seq_len, num_features)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    grad_outputs = torch.ones(critic_interpolates.size(), device=device, requires_grad=False)
    gradients = torch.autograd.grad(outputs=critic_interpolates, inputs=interpolates, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.reshape(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# --- The Heavy Background Task ---
def train_wgan_task(tensor_data: torch.Tensor, latent_dim: int, hidden_dim: int, epochs: int, batch_size: int, lr: float, lambda_gp: float):
    global training_state, models
    
    output_dim = tensor_data.shape[2]
    seq_len = tensor_data.shape[1]
    
    netG = TrafficGenerator(latent_dim, hidden_dim, output_dim).to(device)
    netC = TrafficCritic(output_dim, hidden_dim).to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizerC = optim.Adam(netC.parameters(), lr=lr, betas=(0.0, 0.9))
    
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    training_state["status"] = "training"
    training_state["total_epochs"] = epochs
    
    try:
        for epoch in range(epochs):
            training_state["current_epoch"] = epoch + 1
            epoch_c, epoch_g = [], []
            
            for real_batch, in dataloader:
                real_batch = real_batch.to(device)
                b_size = real_batch.size(0)
                
                # Critic Training
                for _ in range(5):
                    z = torch.randn(b_size, seq_len, latent_dim, device=device)
                    fake_batch = netG(z)
                    gp = compute_gradient_penalty(netC, real_batch, fake_batch.detach(), device=device)
                    loss_C = torch.mean(netC(fake_batch.detach())) - torch.mean(netC(real_batch)) + (lambda_gp * gp)
                    netC.zero_grad()
                    loss_C.backward()
                    optimizerC.step()
                    epoch_c.append(loss_C.item())
                
                # Generator Training
                z = torch.randn(b_size, seq_len, latent_dim, device=device)
                loss_G = -torch.mean(netC(netG(z)))
                netG.zero_grad()
                loss_G.backward()
                optimizerG.step()
                epoch_g.append(loss_G.item())
            
            # Update Global State for Frontend Polling
            training_state["progress"] = (epoch + 1) / epochs
            training_state["c_loss"] = np.mean(epoch_c)
            training_state["g_loss"] = np.mean(epoch_g)
        
        # Save models to memory for generation
        models["netG"] = netG
        models["netC"] = netC
        models["seq_len"] = seq_len
        models["latent_dim"] = latent_dim
        training_state["status"] = "completed"
        
    except Exception as e:
        training_state["status"] = "failed"
        print(f"Training error: {e}")

# --- API Endpoints ---
@app.get("/")
def health_check():
    return {"status": "WGAN-GP API is online", "device": device}

@app.post("/train")
async def start_training(
    background_tasks: BackgroundTasks,
    latent_dim: int = 32,
    hidden_dim: int = 64,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.0002,
    lambda_gp: float = 10.0
    # Note: In a real app, you would pass the preprocessed tensor data here or a file ID. 
    # For simplicity, we assume the frontend sends the processed tensor dimensions.
):
    global training_state
    if training_state["status"] == "training":
        raise HTTPException(status_code=400, detail="Training already in progress.")
    
    # Mocking tensor creation for API independence demo
    # In reality, this endpoint receives the dataset via file upload or shared volume
    dummy_tensor = torch.randn(1000, 10, 79) 
    
    background_tasks.add_task(
        train_wgan_task, dummy_tensor, latent_dim, hidden_dim, epochs, batch_size, lr, lambda_gp
    )
    return {"message": "Training job initiated in background."}

@app.get("/status")
def get_status():
    return training_state

@app.get("/generate")
def generate_traffic(num_samples: int = 1):
    if models["netG"] is None or training_state["status"] != "completed":
        raise HTTPException(status_code=400, detail="Model is not trained yet.")
    
    netG = models["netG"]
    netG.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, models["seq_len"], models["latent_dim"], device=device)
        synthetic_data = netG(z).cpu().numpy()
    
    return {"synthetic_data": synthetic_data.tolist()}
