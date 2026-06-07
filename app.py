import streamlit as st
import numpy as np
import plotly.graph_objects as ui_plots
import time
import torch
import torch.nn as nn
import torch.optim as optim

# --- STAGE 1: SYSTEM & DASHBOARD UI CONFIGURATION ---
st.set_page_config(page_title="CFD vs PINN Dashboard", layout="wide")
st.title("⚡ User-Centric Thermal Simulation: Classical CFD vs. Machine Learning (PINN)")
st.write("An interactive benchmarking framework for the 1D Transient Heat Equation.")

# --- STAGE 2: SIDEBAR INTERACTIVE PARAMETERS ---
st.sidebar.header("🛠️ Simulation Controls")
thermal_diffusivity = st.sidebar.slider("Thermal Diffusivity (α)", 0.01, 0.1, 0.05, step=0.01)
grid_nodes = st.sidebar.slider("Spatial Grid Nodes (Nx)", 10, 50, 20, step=5)
total_time_steps = st.sidebar.slider("Total Time Steps (Nt)", 20, 200, 50, step=10)

st.sidebar.header("🔥 Wall Boundary Conditions")
u_left_input = st.sidebar.slider("Left Boundary Temp (U_left)", 0.0, 5.0, 0.0, step=0.5)
u_right_input = st.sidebar.slider("Right Boundary Temp (U_right)", 0.0, 5.0, 0.0, step=0.5)

st.sidebar.header("🤖 PINN Hyperparameters")
pinn_epochs = st.sidebar.selectbox("Training Epochs", [50, 100, 250, 500, 1000], index=3)

# --- STAGE 3: PYTORCH PINN DEEP LEARNING ARCHITECTURE ---
class ThermalPINN(nn.Module):
    """Multi-Layer Perceptron optimized for automatic differentiation of physical domains."""
    def __init__(self):
        super(ThermalPINN, self).__init__()
        self.input_layer = nn.Linear(2, 32) 
        self.hidden1 = nn.Linear(32, 32)
        self.hidden2 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)
        self.activation = nn.Tanh() # Continuous, infinitely differentiable activation function

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        a1 = self.activation(self.input_layer(inputs))
        a2 = self.activation(self.hidden1(a1))
        a3 = self.activation(self.hidden2(a2))
        return self.output_layer(a3)

def compute_pinn_loss(model, x, t, alpha, X_grid, T_grid, u_L, u_R):
    """Computes multi-headed loss enforcing governing PDEs, initial states, and boundaries."""
    # 1. Governing Heat Equation PDE Loss
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    
    du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
    
    pde_residual = du_dt - alpha * du_dx2
    loss_pde = torch.mean(pde_residual ** 2)
    
    # 2. Initial Condition Loss (T = 0 matching sine wave profile)
    x_raw = np.linspace(0, 1, X_grid.shape[1])
    x_ic = torch.tensor(x_raw, dtype=torch.float32).view(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    u_ic_pred = model(x_ic, t_ic)
    u_ic_true = torch.sin(np.pi * x_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)
    
    # 3. Spatial Boundary Condition Loss (Left and Right Wall enforcement)
    t_raw = np.linspace(0, np.max(T_grid), T_grid.shape[0])
    t_bc = torch.tensor(t_raw, dtype=torch.float32).view(-1, 1)
    x_left = torch.zeros_like(t_bc)
    x_right = torch.ones_like(t_bc)
    
    u_left_pred = model(x_left, t_bc)
    u_right_pred = model(x_right, t_bc)
    loss_bc = torch.mean((u_left_pred - u_L) ** 2) + torch.mean((u_right_pred - u_R) ** 2)
    
    return loss_pde + 10.0 * loss_ic + 10.0 * loss_bc

# --- STAGE 4: DETERMINISTIC CLASSICAL CFD MATHEMATICS ---
def run_classical_cfd(Nx, Nt, alpha, u_L, u_R):
    """Calculates spatial thermal distribution using an explicit FTCS finite difference grid."""
    dx = 1.0 / (Nx - 1)
    dt = 0.1 * (dx**2) / alpha # CFL structural stability criteria check
    
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, Nt * dt, Nt)
    X, T = np.meshgrid(x, t)
    u = np.zeros((Nt, Nx))
    u[0, :] = np.sin(np.pi * x) 
    
    for n in range(0, Nt - 1):
        for i in range(1, Nx - 1):
            u[n+1, i] = u[n, i] + alpha * dt / (dx**2) * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
        u[n+1, 0] = u_L
        u[n+1, -1] = u_R
    return X, T, u

# --- STAGE 5: SYSTEM PIPELINE EXECUTION ---
if st.sidebar.button("🚀 Execute Hybrid Simulation"):
    with st.spinner("Computing classical grids and optimizing physical network weights..."):
        
        # 1. Run Classical Finite Difference Code Block
        start_cfd = time.time()
        X, T, U_cfd = run_classical_cfd(grid_nodes, total_time_steps, thermal_diffusivity, u_left_input, u_right_input)
        cfd_runtime = time.time() - start_cfd
        
        # 2. Run PyTorch Neural Network Optimization Block
        start_pinn = time.time()
        model = ThermalPINN()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1)
        t_tensor = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1)
        
        loss_history = []
        for epoch in range(pinn_epochs):
            optimizer.zero_grad()
            loss = compute_pinn_loss(model, x_tensor, t_tensor, thermal_diffusivity, X, T, u_left_input, u_right_input)
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))
            
        with torch.no_grad():
            u_pred = model(x_tensor, t_tensor)
            U_pinn = u_pred.numpy().reshape(X.shape)
        pinn_runtime = time.time() - start_pinn
        
    # --- STAGE 6: CORE METRICS & DATA DISPLAY VIEW PANEL ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Classical CFD Runtime", f"{cfd_runtime:.5f} sec", "Deterministic Grid")
    col2.metric("PINN Training Runtime", f"{pinn_runtime:.5f} sec", "Meshless Neural Space", delta_color="inverse")
    
    absolute_error = np.abs(U_cfd - U_pinn)
    mean_abs_error = np.mean(absolute_error)
    col3.metric("L2 Physics Discrepancy (MAE)", f"{mean_abs_error:.4f}", "Residual Convergence")
    
    st.subheader("📉 Neural Network Convergence Profile")
    st.line_chart(loss_history)
    st.markdown("---")
    
    view_col1, view_col2 = st.columns(2)
    with view_col1:
        st.subheader("📊 Deterministic CFD Solution Space")
        fig_cfd = ui_plots.Figure(data=[ui_plots.Surface(z=U_cfd, x=X, y=T, colorscale='Viridis')])
        fig_cfd.update_layout(scene=dict(xaxis_title='Spatial (X)', yaxis_title='Time (T)', zaxis_title='Temp (U)'), margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_cfd, use_container_width=True)
        
    with view_col2:
        st.subheader("🧠 Physics-Informed ML Inference Space")
        fig_pinn = ui_plots.Figure(data=[ui_plots.Surface(z=U_pinn, x=X, y=T, colorscale='Cividis')])
        fig_pinn.update_layout(scene=dict(xaxis_title='Spatial (X)', yaxis_title='Time (T)', zaxis_title='Temp (U)'), margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_pinn, use_container_width=True)
        
    st.subheader("📉 Spatial-Temporal Error Distribution Plot")
    fig_err = ui_plots.Figure(data=[ui_plots.Surface(z=absolute_error, x=X, y=T, colorscale='Hot')])
    fig_err.update_layout(scene=dict(xaxis_title='Spatial (X)', yaxis_title='Time (T)', zaxis_title='Absolute Delta'), margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_err, use_container_width=True)
else:
    st.info("💡 Adjust system parameters in the left sidebar configuration panel and click 'Execute Hybrid Simulation' to benchmark calculations.")
