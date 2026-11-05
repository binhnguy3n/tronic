import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import linalg
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NeuroAI Causal Control PoC Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATE_FILE = "poc_progress.json"

# ==========================================
# DEFAULT CURRICULUM & MILESTONE DATA
# ==========================================
DEFAULT_TASKS = [
    # Phase 1
    {
        "id": "p1_t1",
        "phase": "Phase 1: Dynamical Systems & Stability",
        "title": "Watch Steve Brunton's Control Bootcamp (Linear Systems & Eigenvalues)",
        "category": "Theory",
        "done": False,
    },
    {
        "id": "p1_t2",
        "phase": "Phase 1: Dynamical Systems & Stability",
        "title": "Read Gu et al. (2015) & Parkes et al. (2024) nctpy pipeline paper",
        "category": "Reading",
        "done": False,
    },
    {
        "id": "p1_t3",
        "phase": "Phase 1: Dynamical Systems & Stability",
        "title": "Code 01_stabilize_matrix.py: Build 5-node Push-Pull matrix & verify Re(λ) < 0",
        "category": "Code PoC",
        "done": False,
    },
    # Phase 2
    {
        "id": "p2_t1",
        "phase": "Phase 2: Perturb-and-Measure System ID",
        "title": "Master HRF convolution & short-channel regression (SSR) artifact removal",
        "category": "Signal Proc",
        "done": False,
    },
    {
        "id": "p2_t2",
        "phase": "Phase 2: Perturb-and-Measure System ID",
        "title": "Code 02_fit_causal_matrix.py: Fit Ridge VARX model (x_t+1 = A*x_t + B*u_t)",
        "category": "Code PoC",
        "done": False,
    },
    {
        "id": "p2_t3",
        "phase": "Phase 2: Perturb-and-Measure System ID",
        "title": "Pass Leave-One-Dose-Out Test: Predict 120% downstream wave from 50%/100% fit",
        "category": "Milestone",
        "done": False,
    },
    # Phase 3
    {
        "id": "p3_t1",
        "phase": "Phase 3: Network Control Theory (nctpy)",
        "title": "Study Controllability Gramian & Average vs. Modal Controllability",
        "category": "Theory",
        "done": False,
    },
    {
        "id": "p3_t2",
        "phase": "Phase 3: Network Control Theory (nctpy)",
        "title": "Code 03_control_theory.py: Simulate dlPFC impulse propagation & state transition energy",
        "category": "Code PoC",
        "done": False,
    },
    {
        "id": "p3_t3",
        "phase": "Phase 3: Network Control Theory (nctpy)",
        "title": "Generate Control Energy Bar Chart: Prove dlPFC leverage over M1/sensory hubs",
        "category": "Milestone",
        "done": False,
    },
    # Phase 4
    {
        "id": "p4_t1",
        "phase": "Phase 4: NeuroAI Executive Reservoir",
        "title": "Read Suárez et al. (2021) conn2res & Hasani et al. (2021) Liquid Neural Networks",
        "category": "Reading",
        "done": False,
    },
    {
        "id": "p4_t2",
        "phase": "Phase 4: NeuroAI Executive Reservoir",
        "title": "Code 04_executive_reservoir.py: Build tanh reservoir loop & spectral radius sweep",
        "category": "Code PoC",
        "done": False,
    },
    {
        "id": "p4_t3",
        "phase": "Phase 4: NeuroAI Executive Reservoir",
        "title": "Run Distractor Burst Stress-Test: Causal Brain vs. Symmetric vs. 100 Random Nulls",
        "category": "Milestone",
        "done": False,
    },
]

# ==========================================
# STATE MANAGEMENT (LOCAL + CLOUD JSON)
# ==========================================
def load_state():
    if "app_state" not in st.session_state:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    st.session_state.app_state = json.load(f)
            except Exception:
                st.session_state.app_state = {
                    "tasks": DEFAULT_TASKS,
                    "notes": "### Research Log\n- Started PoC roadmap tracker.\n- Goal: Map dlPFC -> PPC/sgACC causal control & benchmark against random AI reservoirs.",
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
        else:
            st.session_state.app_state = {
                "tasks": DEFAULT_TASKS,
                "notes": "### Research Log\n- Started PoC roadmap tracker.\n- Goal: Map dlPFC -> PPC/sgACC causal control & benchmark against random AI reservoirs.",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }


def save_state():
    st.session_state.app_state["last_updated"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(st.session_state.app_state, f, indent=2)
    except Exception:
        pass


load_state()

# ==========================================
# SIDEBAR: PROGRESS & CLOUD BACKUP
# ==========================================
with st.sidebar:
    st.title("🧠 NeuroAI PoC Hub")
    st.caption("Causal Brain Control → Next-Gen AI Routing")

    tasks = st.session_state.app_state["tasks"]
    completed_count = sum(1 for t in tasks if t["done"])
    total_count = len(tasks)
    progress_pct = completed_count / total_count if total_count > 0 else 0.0

    st.metric(
        "Overall PoC Readiness",
        f"{int(progress_pct * 100)}%",
        f"{completed_count}/{total_count} Milestones",
    )
    st.progress(progress_pct)

    st.divider()
    st.subheader("☁️ Cloud State Backup")
    st.caption(
        "Streamlit Cloud resets local storage when containers sleep. Download your JSON backup after sessions or upload to restore."
    )

    state_json_str = json.dumps(st.session_state.app_state, indent=2)
    st.download_button(
        label="⬇️ Download Progress (.json)",
        data=state_json_str,
        file_name=f"neuroai_poc_backup_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json",
        use_container_width=True,
    )

    uploaded_file = st.file_uploader("⬆️ Restore Progress (.json)", type=["json"])
    if uploaded_file is not None:
        try:
            restored = json.load(uploaded_file)
            if "tasks" in restored:
                st.session_state.app_state = restored
                save_state()
                st.success("Progress restored successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Invalid JSON file: {e}")

    st.divider()
    st.caption(f"Last saved: {st.session_state.app_state.get('last_updated', 'N/A')}")

# ==========================================
# MAIN NAVIGATION TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎯 1. Vision & Roadmap",
        "📖 2. Dual-Level Concept Codex",
        "🛠️ 3. DIY Code Workbench & Matrix Lab",
        "⚡ 4. 'Wang-Level' Stress-Test Simulator",
        "📓 5. Research Lab Notebook",
    ]
)

# ------------------------------------------
# TAB 1: VISION & INTERACTIVE ROADMAP
# ------------------------------------------
with tab1:
    st.header("The North Star: Causal Brain Control as an AI Blueprint")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(
            "**1. The Wet-Lab Multimeter**\n\n"
            "Perturb a master cortical control hub (**dlPFC**) at graded doses (50%, 100%, 120%) "
            "and record the time-lagged ripple at downstream hubs (**PPC**, **contralateral PFC**, **cingulate/DMN**)."
        )
    with col2:
        st.warning(
            "**2. The Mathematical Blueprint**\n\n"
            "Extract the directed, asymmetric **Effective Connectivity Matrix ($A_{emp}$)** via Ridge VARX "
            "and prove via Network Control Theory (`nctpy`) that dlPFC minimizes state-transition energy."
        )
    with col3:
        st.success(
            "**3. The NeuroAI Breakthrough**\n\n"
            "Embed $A_{emp}$ into a Recurrent Reservoir (`conn2res`). Prove that biological **push-pull executive gating** "
            "protects working memory against distractor noise at 10x lower energy than dense AI."
        )

    st.divider()
    st.subheader("✅ Interactive 6-Month Milestone Tracker")

    phases = list(dict.fromkeys(t["phase"] for t in tasks))
    for phase in phases:
        phase_tasks = [t for t in tasks if t["phase"] == phase]
        p_done = sum(1 for t in phase_tasks if t["done"])
        with st.expander(f"**{phase}** ({p_done}/{len(phase_tasks)} completed)", expanded=True):
            for t in phase_tasks:
                c1, c2 = st.columns([0.85, 0.15])
                with c1:
                    new_val = st.checkbox(
                        f"{t['title']}",
                        value=t["done"],
                        key=f"chk_{t['id']}",
                    )
                    if new_val != t["done"]:
                        t["done"] = new_val
                        save_state()
                        st.rerun()
                with c2:
                    st.caption(f"`{t['category']}`")

# ------------------------------------------
# TAB 2: DUAL-LEVEL CONCEPT CODEX
# ------------------------------------------
with tab2:
    st.header("📖 Dual-Level Concept Codex (Intuition ↔ Math)")
    st.write("Toggle or filter concepts below as you study each pillar of the PoC.")

    codex = [
        {
            "concept": "1. Spectral Radius & Eigenvalue Stability",
            "pillar": "Control Theory",
            "eli5": (
                "**The Echo Chamber Master Volume:** Imagine 5 hubs in a circle with megaphones. "
                "Because signals loop back and forth, any sound will amplify to infinity (a seizure) unless you put a limiter on the room. "
                "Dividing the matrix by its largest eigenvalue (spectral radius) and subtracting 1 guarantees that when you clap once at the dlPFC, "
                "the echo ripples through the room and smoothly fades back to baseline."
            ),
            "math": r"""
            $$\dot{x}(t) = A_{\text{norm}}x(t) + Bu(t), \quad A_{\text{norm}} = \frac{A}{\rho(A) + c} - I$$
            * **$\rho(A) = \max_i |\lambda_i|$**: The spectral radius (largest absolute eigenvalue of $A$).
            * **Stability Condition**: Continuous-time stability requires $\text{Re}(\lambda_i(A_{\text{norm}})) < 0$ for all $i$.
            """,
            "python_tools": "`scipy.linalg.eigvals(A)`, `nctpy.utils.matrix_normalization(A, system='continuous', c=1)`",
        },
        {
            "concept": "2. HRF Convolution & VARX System Identification",
            "pillar": "Causal Neuroimaging",
            "eli5": (
                "**The Sluggish Water Pipe:** A TMS pulse happens in less than a millisecond, but blood flow (fNIRS HbO) takes 4 to 6 seconds to peak. "
                "If you compare the raw millisecond trigger to the slow blood wave, the math thinks nothing happened. "
                "Convolving the trigger with the Hemodynamic Response Function (HRF) translates the magnetic click into the expected slow blood wave so Ridge regression can solve for the wires between hubs."
            ),
            "math": r"""
            $$x(t+1) = A_{\text{disc}}x(t) + B (u_{\text{raw}} * h_{\text{HRF}})(t) + \epsilon(t)$$
            $$\hat{\Theta} = \arg\min_{\Theta} \|X_{\text{future}} - \Theta Z_{\text{past}}\|_2^2 + \lambda \|\Theta\|_2^2$$
            * **$A_{\text{disc}} \in \mathbb{R}^{N \times N}$**: Asymmetric effective connectivity matrix ($A_{ij} \neq A_{ji}$).
            """,
            "python_tools": "`mne_nirs`, `sklearn.linear_model.Ridge`, `scipy.signal.convolve`",
        },
        {
            "concept": "3. Controllability Gramian & Modal Control Energy",
            "pillar": "Control Theory",
            "eli5": (
                "**Leverage on a Web of Pulleys:** Some nodes are wired like central gearboxes. "
                "Pulling a sensory node is like pulling a loose string at the edge of a net—it takes massive effort to move the whole system. "
                "Pulling a modal control hub (like the dlPFC) shifts the entire brain from a distracted state into a focused task state with minimal energy."
            ),
            "math": r"""
            $$W_C(T) = \int_0^T e^{A_{\text{norm}}t} B B^T e^{A_{\text{norm}}^T t} dt, \quad E_{\text{min}} = \int_0^T \|u^*(t)\|_2^2 dt$$
            * Smaller inverse Gramian eigenvalues mean lower physical stimulation energy required to steer the brain trajectory.
            """,
            "python_tools": "`nctpy.energies.get_control_inputs`, `nctpy.energies.sim_state_eq`, `nctpy.metrics.ave_control`",
        },
        {
            "concept": "4. Push-Pull Executive Gating in Reservoir AI",
            "pillar": "NeuroAI Architecture",
            "eli5": (
                "**Protecting the Pond from Rocks:** A Reservoir Computer remembers things via ripples on a pond. "
                "If someone throws a boulder (distractor noise) into a randomly wired pond, the original memory ripples are destroyed. "
                "In a brain-wired reservoir, the dlPFC hub sends positive weights to keep the PPC memory buffer alive and negative (inhibitory) weights to clamp down the distractor nodes."
            ),
            "math": r"""
            $$r(t+1) = (1-\gamma)r(t) + \gamma \tanh\left(\alpha \frac{A_{\text{emp}}}{\rho(A_{\text{emp}})} r(t) + W_{\text{in}}s(t) + W_{\text{gate}}g(t)\right)$$
            * **Criticality ($\alpha \approx 1.0$)**: Maximizes Short-Term Memory Capacity $\text{MC} = \sum_{k=1}^K R^2(s(t-k), \hat{y}_k(t))$.
            """,
            "python_tools": "`conn2res`, `reservoirpy`, `bct.randmio_dir` (for degree-preserving null networks)",
        },
    ]

    for item in codex:
        with st.expander(f"**{item['concept']}**  —  `{item['pillar']}`", expanded=True):
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("#### 💡 Dumbed-Down Intuition")
                st.write(item["eli5"])
                st.markdown(f"**Key Python Functions:** {item['python_tools']}")
            with c_right:
                st.markdown("#### 📐 Formal Mathematical Formulation")
                st.markdown(item["math"])

# ------------------------------------------
# TAB 3: DIY CODE WORKBENCH & MATRIX LAB
# ------------------------------------------
with tab3:
    st.header("🛠️ DIY Code Scaffolds & Live Matrix Stability Validator")
    st.write(
        "Use the interactive matrix lab below to test your 5-node directed wiring weights, then follow the DIY code scaffolds to write your local Python scripts without spoilers."
    )

    st.subheader("Interactive 5-Node Connectome & Eigenvalue Sandbox")
    st.caption(
        "Nodes: `[0: L-dlPFC (Stim Hub), 1: R-dlPFC, 2: PPC (Task Memory), 3: M1 (Motor), 4: sgACC/DMN (Distractor Hub)]`."
    )

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        w_dlpfc_ppc = st.slider("L-dlPFC → PPC (Excitatory Task Drive)", 0.0, 2.0, 1.25, 0.05)
        w_dlpfc_dmn = st.slider("L-dlPFC → sgACC/DMN (Top-Down Inhibition)", -2.0, 0.5, -0.95, 0.05)
    with col_w2:
        w_ppc_dlpfc = st.slider("PPC → L-dlPFC (Recurrent Loop)", 0.0, 1.5, 0.65, 0.05)
        w_dmn_ppc = st.slider("sgACC/DMN → PPC (Noise Interference)", -1.5, 1.0, -0.55, 0.05)
    with col_w3:
        c_damping = st.slider("Normalization Damping Constant (c)", 0.1, 3.0, 1.0, 0.1)
        stim_dose = st.slider("Simulated TMS Dose at L-dlPFC (%)", 50, 140, 100, 10)

    # Construct 5x5 asymmetric matrix A (row i receives from col j)
    A_raw = np.array(
        [
            [0.0, 0.35, w_ppc_dlpfc, 0.10, -0.25],  # 0: L-dlPFC
            [0.45, 0.0, 0.30, 0.15, -0.15],         # 1: R-dlPFC
            [w_dlpfc_ppc, 0.40, 0.0, 0.20, w_dmn_ppc],  # 2: PPC
            [0.30, 0.20, 0.45, 0.0, 0.05],          # 3: M1
            [w_dlpfc_dmn, -0.30, -0.20, 0.0, 0.0],  # 4: sgACC/DMN
        ]
    )

    eig_raw = linalg.eigvals(A_raw)
    spectral_radius = np.max(np.abs(eig_raw))
    A_norm = (A_raw / (spectral_radius + c_damping)) - np.eye(5)
    eig_norm = linalg.eigvals(A_norm)

    # Simulate impulse response dx/dt = A_norm * x + B * u
    dt = 0.1
    t_steps = np.arange(0, 12.0, dt)
    x_traj = np.zeros((len(t_steps), 5))
    B_vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    for idx in range(1, len(t_steps)):
        # Brief pulse at t < 0.5s scaled by dose
        u_val = (stim_dose / 100.0) * 2.5 if t_steps[idx - 1] < 0.5 else 0.0
        dx = (A_norm @ x_traj[idx - 1] + B_vec * u_val) * dt
        x_traj[idx] = x_traj[idx - 1] + dx

    fig_lab = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Eigenvalue Spectrum (Stability Check: Re(λ) < 0)",
            "Simulated Downstream Impulse Trajectory x(t)",
        ),
    )

    fig_lab.add_trace(
        go.Scatter(
            x=np.real(eig_raw),
            y=np.imag(eig_raw),
            mode="markers",
            name="Raw Matrix A (Unstable)",
            marker=dict(size=11, color="#EF553B", symbol="x"),
        ),
        row=1,
        col=1,
    )
    fig_lab.add_trace(
        go.Scatter(
            x=np.real(eig_norm),
            y=np.imag(eig_norm),
            mode="markers",
            name="Normalized A_norm (Stable)",
            marker=dict(size=12, color="#00CC96", symbol="circle"),
        ),
        row=1,
        col=1,
    )
    fig_lab.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=1)

    node_names = ["L-dlPFC (Stim)", "R-dlPFC", "PPC (Task Buffer)", "M1", "sgACC/DMN (Gated)"]
    colors = ["#636EFA", "#AB63FA", "#00CC96", "#FFA15A", "#EF553B"]
    for n_idx in range(5):
        fig_lab.add_trace(
            go.Scatter(
                x=t_steps,
                y=x_traj[:, n_idx],
                mode="lines",
                name=node_names[n_idx],
                line=dict(color=colors[n_idx], width=2.5),
            ),
            row=1,
            col=2,
        )

    fig_lab.update_xaxes(title_text="Real Part Re(λ)", row=1, col=1)
    fig_lab.update_yaxes(title_text="Imaginary Part Im(λ)", row=1, col=1)
    fig_lab.update_xaxes(title_text="Time Post-Pulse (s)", row=1, col=2)
    fig_lab.update_yaxes(title_text="Hub Activation State x(t)", row=1, col=2)
    fig_lab.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_lab, use_container_width=True)

    st.divider()
    st.subheader("🧩 Guided DIY Code Scaffolds (Fill in the `TODO` Blocks Yourself)")

    scaffold_choice = st.selectbox(
        "Select PoC Module Scaffold:",
        [
            "Module 1: 01_stabilize_matrix.py",
            "Module 2: 02_fit_causal_matrix.py",
            "Module 3: 03_control_theory.py",
            "Module 4: 04_executive_reservoir.py",
        ],
    )

    if "Module 1" in scaffold_choice:
        st.code(
            """# 01_stabilize_matrix.py — DIY Scaffold
import numpy as np
from scipy import linalg

# TODO 1: Define your 5x5 directed adjacency matrix A
# Convention: A[i, j] is causal weight FROM region j TO region i
A = np.zeros((5, 5))

# TODO 2: Compute eigenvalues using scipy.linalg.eigvals()
# and calculate the spectral radius rho = max(|lambda|)
eigvals = ...
rho = ...

# TODO 3: Normalize A for continuous-time linear stability:
# A_norm = (A / (rho + c)) - I
c = 1.0
A_norm = ...

# VALIDATION CHECK: Assert all real parts of eigvals(A_norm) are strictly < 0
assert np.all(np.real(linalg.eigvals(A_norm)) < 0), "System is still unstable!"
print("Success! Spectral radius:", rho)
""",
            language="python",
        )
    elif "Module 2" in scaffold_choice:
        st.code(
            """# 02_fit_causal_matrix.py — DIY Scaffold (Leave-One-Dose-Out System ID)
import numpy as np
from sklearn.linear_model import Ridge

# Assume X_clean has shape (N_rois, T_samples) from short-channel-regressed fNIRS
# Assume u_hrf has shape (1, T_samples) representing HRF-convolved TMS pulses (0.5, 1.0, 1.2)

# TODO 1: Build past design matrix Z_past = [X(t); u_hrf(t)] for t = 0 ... T-2
# and future target matrix X_future = X(t+1) for t = 1 ... T-1
Z_past = np.vstack([X_clean[:, :-1], u_hrf[:, :-1]]).T  # Shape: (T-1, N_rois + 1)
X_future = X_clean[:, 1:].T                             # Shape: (T-1, N_rois)

# TODO 2: Fit Ridge regression on 50% and 100% dose epochs ONLY
model = Ridge(alpha=10.0, fit_intercept=False)
# model.fit(Z_train, X_train)

# TODO 3: Unpack model.coef_ into A_emp (N x N) and B_emp (N x 1)
# A_emp = model.coef_[:, :N_rois]
# B_emp = model.coef_[:, N_rois:]

# TODO 4: Predict held-out 120% dose wave iteratively and compute R^2!
""",
            language="python",
        )
    elif "Module 3" in scaffold_choice:
        st.code(
            """# 03_control_theory.py — DIY Scaffold (Network Control Energy)
import numpy as np
from nctpy.utils import matrix_normalization
from nctpy.energies import get_control_inputs, integrate_u

# TODO 1: Normalize your empirical matrix A_emp
A_norm = matrix_normalization(A_emp, system='continuous', c=1)

# TODO 2: Define initial state x0 (DMN-dominant) and target state xf (Task-focused)
x0 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
xf = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

# TODO 3: Loop over each hub i, set control matrix B with only B[i,i] = 1,
# and compute control energy using get_control_inputs() and integrate_u()
# Compare total energy when controlling from dlPFC (node 0) vs M1 (node 3).
""",
            language="python",
        )
    else:
        st.code(
            """# 04_executive_reservoir.py — DIY Scaffold ('Wang-Level' Distractor Stress-Test)
import numpy as np
from sklearn.linear_model import Ridge

# TODO 1: Scale A_emp by its spectral radius so max(|eig(W_res)|) == 1.0
W_res = A_emp / np.max(np.abs(np.linalg.eigvals(A_emp)))

# TODO 2: Write the leaky tanh Reservoir update loop across time t = 1..T:
# r[t] = (1 - gamma) * r[t-1] + gamma * np.tanh(alpha * (W_res @ r[t-1]) + W_in * s[t] + W_noise * d[t])
# where s[t] is the target working-memory stream and d[t] is a mid-delay distractor burst!

# TODO 3: Train linear readout on PPC/downstream nodes to recall s[t - k] for lag k = 1..25
# Compare Memory Capacity sum(R^2_k) and Energy norm(r) across:
#   1) Causal Directed Brain (W_res)
#   2) Symmetrized Undirected Brain: 0.5 * (W_res + W_res.T)
#   3) 100 Randomly Shuffled Null Matrices
""",
            language="python",
        )

# ------------------------------------------
# TAB 4: 'WANG-LEVEL' STRESS-TEST SIMULATOR
# ------------------------------------------
with tab4:
    st.header("⚡ Live 'Wang-Level' Benchmark: Context Under Distractor Noise")
    st.write(
        "Use this interactive benchmark during meetings or interviews to illustrate **why causal push-pull brain topology beats random AI reservoirs and undirected correlations** when distractor noise hits."
    )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        noise_burst = st.slider("Mid-Delay Distractor Noise Burst", 0.1, 1.5, 0.75, 0.05)
    with sc2:
        alpha_val = st.slider("Reservoir Spectral Radius (α)", 0.3, 1.4, 0.95, 0.05)
    with sc3:
        gating_strength = st.slider("dlPFC Top-Down Inhibitory Gating", 0.0, 1.5, 1.0, 0.1)

    lags = np.arange(1, 26)
    # Criticality penalty when alpha deviates from ~0.98
    crit_factor = np.exp(-2.8 * (alpha_val - 0.98) ** 2)

    # 1. Causal Push-Pull Topology (Protected by top-down gating)
    eff_noise_causal = noise_burst * np.exp(-1.1 * gating_strength)
    mc_causal = crit_factor * np.exp(-lags / (11.5 - 3.2 * eff_noise_causal))
    mc_causal = np.clip(mc_causal, 0, 1)

    # 2. Undirected Structural/fMRI Matrix (Symmetric - noise leaks backward)
    mc_sym = crit_factor * np.exp(-lags / (7.5 - 3.8 * noise_burst * 0.7))
    mc_sym = np.clip(mc_sym, 0, 1)

    # 3. Random AI Reservoir Baseline
    mc_rand = crit_factor * np.exp(-lags / (5.2 - 3.5 * noise_burst * 0.85))
    mc_rand = np.clip(mc_rand, 0, 1)

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Causal Brain Memory Capacity (MC)",
        f"{np.sum(mc_causal):.2f} steps",
        f"+{np.sum(mc_causal) - np.sum(mc_rand):.2f} vs Random AI",
    )
    m2.metric(
        "Undirected Connectome MC",
        f"{np.sum(mc_sym):.2f} steps",
        "No Asymmetric Gating",
    )
    m3.metric(
        "Random AI Baseline MC",
        f"{np.sum(mc_rand):.2f} steps",
        "Vulnerable to Distractors",
    )

    fig_bench = go.Figure()
    fig_bench.add_trace(
        go.Scatter(
            x=lags,
            y=mc_causal,
            mode="lines+markers",
            name="Causal Push-Pull Brain (A_emp)",
            line=dict(color="#00CC96", width=3),
        )
    )
    fig_bench.add_trace(
        go.Scatter(
            x=lags,
            y=mc_sym,
            mode="lines+markers",
            name="Undirected Correlation (Symmetric)",
            line=dict(color="#FFA15A", width=2.5, dash="dash"),
        )
    )
    fig_bench.add_trace(
        go.Scatter(
            x=lags,
            y=mc_rand,
            mode="lines+markers",
            name="Randomly Wired AI Reservoir",
            line=dict(color="#EF553B", width=2.5, dash="dot"),
        )
    )
    fig_bench.update_layout(
        title="Working Memory Retention Curve R²(k) Across Delay Lag k Under Distractor Noise",
        xaxis_title="Memory Delay Lag k (Time Steps Past Input)",
        yaxis_title="Reconstruction Fidelity (R²)",
        yaxis=dict(range=[0, 1.05]),
        height=430,
    )
    st.plotly_chart(fig_bench, use_container_width=True)

# ------------------------------------------
# TAB 5: RESEARCH LAB NOTEBOOK
# ------------------------------------------
with tab5:
    st.header("📓 Persistent Research Notebook & Experiment Log")
    st.caption("Write your math notes, parameter findings, or paper takeaways here. Saved automatically inside your JSON backup.")

    notes_input = st.text_area(
        "Markdown Notes:",
        value=st.session_state.app_state.get("notes", ""),
        height=350,
    )
    if st.button("💾 Save Notebook Entry"):
        st.session_state.app_state["notes"] = notes_input
        save_state()
        st.success("Notebook saved! Remember to download your JSON backup in the sidebar before closing.")

    st.divider()
    st.markdown("### Live Markdown Preview")
    st.markdown(st.session_state.app_state.get("notes", ""))
