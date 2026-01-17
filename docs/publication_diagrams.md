# OpenAD-lib Publication Diagrams (Mermaid)

## Diagram 1: Technology Stack Architecture

```mermaid
graph TB
    subgraph " "
        OPT["<b>Optimisation/Mechanistic Models</b>"]
        ML["<b>ML/Surrogate</b>"]
        FEED["<b>Feedstock Library</b>"]
        MPC["<b>MPC Control</b>"]
    end
    
    subgraph " "
        SCIPY["<b>Pyomo/SciPy</b>"]
        TORCH["<b>PyTorch</b>"]
        NUMPY["<b>NumPy/SciPy</b>"]
        CASADI["<b>CasADi/do-mpc</b>"]
    end
    
    subgraph " "
        INTEG["<b>Integration & Control Layer</b>"]
    end
    
    OPT --> SCIPY
    ML --> TORCH
    FEED --> NUMPY
    MPC --> CASADI
    
    SCIPY --> INTEG
    TORCH --> INTEG
    NUMPY --> INTEG
    CASADI --> INTEG
    
    classDef topBox fill:#E6E6FA,stroke:#4B0082,stroke-width:2px,color:#000
    classDef techBox fill:#B0C4DE,stroke:#4682B4,stroke-width:2px,color:#000
    classDef mlBox fill:#90EE90,stroke:#228B22,stroke-width:2px,color:#000
    classDef feedBox fill:#F5DEB3,stroke:#DAA520,stroke-width:2px,color:#000
    classDef mpcBox fill:#FFB6C1,stroke:#C71585,stroke-width:2px,color:#000
    classDef integBox fill:#FFFACD,stroke:#FFD700,stroke-width:3px,color:#000
    
    class OPT topBox
    class ML mlBox
    class FEED feedBox
    class MPC mpcBox
    class SCIPY,TORCH,NUMPY,CASADI techBox
    class INTEG integBox
```

### Key Components:
- **Pyomo/SciPy**: ADM1, reduced models, optimization formulations
- **PyTorch**: ANNs, LSTMs, MTGP, PINNs, differentiable programming
- **NumPy/SciPy**: Feedstock descriptors, distributions, ACoD processing
- **CasADi/do-mpc**: Nonlinear MPC, constraint handling, real-time control
- **Integration**: Seamless hybrid workflows with uncertainty propagation

---

## Diagram 2: Simplified Component Overview

```mermaid
graph TB
    FEEDSTOCK["<b>Feedstock Library</b><br/>• Descriptors<br/>• Distributions<br/>• ACoD Generator"]
    MECH["<b>Mechanistic Models</b><br/>• ADM1 (38-state)<br/>• AM2 (4-state)"]
    ML["<b>ML/Surrogates</b><br/>• MTGP<br/>• LSTM<br/>• PINNs (planned)<br/>• DMD (planned)"]
    OPT["<b>Optimisation & Control</b><br/>• Optuna Calibration<br/>• Bayesian Opt (planned)<br/>• MPC Control<br/>• Scheduling (planned)"]
    
    INTEGRATION["<b>OpenAD-lib Integration Layer</b><br/>Data • Validation • Plotting • Metrics • API"]
    
    FEEDSTOCK --> INTEGRATION
    MECH --> INTEGRATION
    ML --> INTEGRATION
    OPT --> INTEGRATION
    
    classDef compBox fill:#6495ED,stroke:#4169E1,stroke-width:2px,color:#FFF
    classDef integBox fill:#FFD700,stroke:#FFA500,stroke-width:3px,color:#000
    
    class FEEDSTOCK,MECH,ML,OPT compBox
    class INTEGRATION integBox
```

---

## Diagram 3: Detailed Technology Stack (4 Pillars)

```mermaid
graph TB
    subgraph Layer1[" "]
        direction LR
        A["<b>Optimisation/<br/>Mechanistic Models</b>"]
        B["<b>ML/Surrogate</b>"]
        C["<b>Feedstock Library</b>"]
        D["<b>MPC Control</b>"]
    end
    
    subgraph Layer2[" "]
        direction LR
        A1["<b>Pyomo/SciPy</b>"]
        B1["<b>PyTorch/GPyTorch</b>"]
        C1["<b>NumPy/SciPy</b>"]
        D1["<b>CasADi/do-mpc</b>"]
    end
    
    subgraph Layer3[" "]
        INT["<b>Integration & Control Layer</b>"]
    end
    
    A --> A1
    B --> B1
    C --> C1
    D --> D1
    
    A1 --> INT
    B1 --> INT
    C1 --> INT
    D1 --> INT
    
    classDef optBox fill:#E6E6FA,stroke:#4B0082,stroke-width:2px,color:#000
    classDef mlBox fill:#90EE90,stroke:#228B22,stroke-width:2px,color:#000
    classDef feedBox fill:#F5DEB3,stroke:#DAA520,stroke-width:2px,color:#000
    classDef mpcBox fill:#FFB6C1,stroke:#C71585,stroke-width:2px,color:#000
    classDef integBox fill:#FFFACD,stroke:#FFD700,stroke-width:3px,color:#000
    
    class A,A1 optBox
    class B,B1 mlBox
    class C,C1 feedBox
    class D,D1 mpcBox
    class INT integBox
```

---

## Usage Notes

**For LaTeX/Papers**:
- Copy Mermaid code into [Mermaid Live Editor](https://mermaid.live)
- Export as PNG/SVG with transparent background
- Insert into LaTeX using `\includegraphics{}`

**For Markdown/GitHub**:
- Paste Mermaid code directly - renders automatically
- Works in README.md, documentation, GitHub Issues

**For Presentations**:
- Export as high-resolution PNG from Mermaid Live
- Use SVG for vector graphics (scales perfectly)

**Color Scheme**:
- Purple: Optimisation/Mechanistic (SciPy-based)
- Green: ML/Surrogates (PyTorch-based)  
- Beige: Feedstock/Data (NumPy-based)
- Pink: MPC Control (CasADi-based)
- Yellow: Integration Layer

---

## Customization Tips

### Change Colors
```mermaid
classDef myClass fill:#COLOR,stroke:#BORDER,stroke-width:2px
class NodeName myClass
```

### Add More Details
```mermaid
NODE["<b>Title</b><br/>• Item 1<br/>• Item 2"]
```

### Adjust Layout
```
TB (top-to-bottom)
LR (left-to-right)  
RL (right-to-left)
BT (bottom-to-top)
```
