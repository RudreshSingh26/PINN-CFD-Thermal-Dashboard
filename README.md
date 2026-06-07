# Hybrid Scientific Computing Dashboard: Classical CFD vs. Physics-Informed Neural Networks (PINNs)

An interactive, user-centric scientific computing dashboard that benchmarks traditional finite-difference numerical solvers against state-of-the-art deep learning frameworks (PINNs) for transient heat transfer applications.

🚀 **Live Interactive Application:** [Paste Your Streamlit Cloud App URL Here]

---

## 📊 Core Engineering Architecture

This platform simulates and solves the **1D Transient Heat Equation** governed by the following partial differential equation (PDE):

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

The system models a solid domain subjected to a sinusoidal initial temperature distribution ($U = \sin(\pi x)$) and dynamic, user-controlled asymmetric Dirichlet boundary conditions at the outer walls ($X=0$ and $X=1$).

### 1. Deterministic Numerical Solver (Classical CFD)
* **Discretization Scheme:** Forward-Time Central-Space (FTCS) explicit finite-difference grid framework.
* **Stability Constraint:** Enforces strict structural convergence checking via the Courant–Friedrichs–Lewy (CFL) numerical safety condition:
  $$\Delta t = 0.1 \cdot \frac{\Delta x^2}{\alpha}$$

### 2. Deep Learning Solver (Physics-Informed Neural Network)
* **Topology:** 3-layer deep Multi-Layer Perceptron (MLP) mapping continuous coordinates $(x, t) \rightarrow u$.
* **Activation Operator:** Infinitely differentiable Hyperbolic Tangent ($\text{Tanh}$) curves designed to compute clean calculus gradients.
* **Loss Compilation:** Employs exact Automatic Differentiation (torch.autograd.grad) bypassing rigid space-time geometric grids via a composite multi-headed loss profile: 

Total Loss = Loss_PDE + 10 * Loss_Initial + 10 * Loss_Boundary


---

## 🛠️ Dashboard Interactive Capabilities

* **Dynamic Boundary Physics:** Live sidebar sliders allowing users to heat or cool left ($U_{\text{left}}$) and right ($U_{\text{right}}$) domain walls instantly deforming the physical solutions.
* **Live Optimizer Diagnostics:** Real-time line chart tracking loss convergence history profiles as the PyTorch Adam optimization algorithm loops through training epochs.
* **Spatial-Temporal Error Mapping:** Fully interactive 3D Plotly surface graphics mapping real-time $L_2$ Mean Absolute Error (MAE) discrepancies across continuous time domains.

---

## 📂 Repository File Structure

* `app.py`: Cleaned, production-ready full-stack python dashboard script linking core backend physics loops with the web layout.
* `requirements.txt`: System dependencies package manifest for automated cloud runtime configurations.

---

## 🎓 Academic Framework & Contact

This optimization project was designed and engineered as a pre-term portfolio exploration ahead of entering the **Thermal Stream (Department of Mechanical Engineering) at the Indian Institute of Science (IIT) Madras**. 

* **Developer:** [Your Name]
* **LinkedIn:** [Your LinkedIn URL]
* **Target Domain:** Physics-ML, Automated Scientific Workflows, Surrogate Modeling, AI-Accelerated CFD.
