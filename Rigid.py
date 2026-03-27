นผิวถนนแข็งตามมาตรฐาน AASHTO 1993 · PY

สำเนา

"""
AASHTO 1993 Rigid Pavement Design — Concrete Pavement
ออกแบบผิวทางคอนกรีต ตามวิธี AASHTO 1993
กรมทางหลวง - Department of Highways Thailand
"""
 
import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
import pandas as pd
from scipy.optimize import brentq
 
# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AASHTO 1993 | ออกแบบผิวทางคอนกรีต",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
 
html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
h1,h2,h3 { font-family: 'Sarabun', sans-serif; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }
 
.header-box {
    background: linear-gradient(135deg, #1a2e4a 0%, #0d1f33 60%, #0a1520 100%);
    border: 1px solid #2d5a8a; border-radius: 12px;
    padding: 24px 32px; margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(45,90,138,0.35);
}
.header-box h1 { color: #e6f3ff; font-size: 1.8rem; font-weight: 700; margin: 0 0 4px 0; }
.header-box p  { color: #7ab3d4; margin: 0; font-size: 0.95rem; }
 
.result-card {
    background: linear-gradient(135deg, #1a2e1a 0%, #0f1f0f 100%);
    border: 1px solid #2d6b2d; border-radius: 10px;
    padding: 18px; text-align: center; margin-bottom: 8px;
}
.result-card .label { color: #7dd87d; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; }
.result-card .value { color: #c8f5c8; font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem; font-weight: 700; }
.result-card .unit  { color: #7dd87d; font-size: 0.8rem; }
 
.result-card-blue {
    background: linear-gradient(135deg, #1a2a3e 0%, #0f1a2d 100%);
    border: 1px solid #2d5a8a; border-radius: 10px;
    padding: 18px; text-align: center; margin-bottom: 8px;
}
.result-card-blue .label { color: #7ab3d4; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; }
.result-card-blue .value { color: #b3d9f5; font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem; font-weight: 700; }
.result-card-blue .unit  { color: #7ab3d4; font-size: 0.8rem; }
 
.result-card-orange {
    background: linear-gradient(135deg, #2e1a0a 0%, #1f1005 100%);
    border: 1px solid #8a5a20; border-radius: 10px;
    padding: 18px; text-align: center; margin-bottom: 8px;
}
.result-card-orange .label { color: #d4a044; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; }
.result-card-orange .value { color: #f5d07a; font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem; font-weight: 700; }
.result-card-orange .unit  { color: #d4a044; font-size: 0.8rem; }
 
.section-header {
    background: linear-gradient(90deg, #1e3a5f 0%, transparent 100%);
    border-left: 4px solid #0d74e7;
    padding: 8px 14px; border-radius: 0 8px 8px 0; margin: 14px 0 10px 0;
}
.section-header h3 { color: #a8d0f0; margin: 0; font-size: 0.95rem; font-weight: 600; }
 
.layer-box {
    border-radius: 6px; padding: 12px 16px; margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
    display: flex; justify-content: space-between; align-items: center;
}
.layer-concrete  { background: #1a1e2e; border: 2px solid #4a6fa8; color: #a8c8f0; }
.layer-base      { background: #0d1a0d; border: 1px solid #4a7a4a; color: #8ac88a; }
.layer-subbase   { background: #0f0f1a; border: 1px solid #4a4a8a; color: #8a8ac8; }
.layer-subgrade  { background: #1a0f0a; border: 1px solid #8a5a3a; color: #c8a07a; }
 
.formula-box {
    background: #0a0f1a; border: 1px solid #1e3a5f; border-radius: 8px;
    padding: 16px; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; color: #7ab3d4; line-height: 2.0;
}
.info-box {
    background: #111d2e; border: 1px solid #1e3a5f; border-radius: 8px;
    padding: 16px; font-size: 0.85rem; color: #8aafcc; line-height: 1.7;
}
.check-ok   { color: #4caf50; font-weight: 700; }
.check-fail { color: #f44336; font-weight: 700; }
 
[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e3a5f; }
.stTabs [data-baseweb="tab-list"] { background: #111d2e; border-radius: 8px; padding: 4px; }
</style>
""", unsafe_allow_html=True)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# AASHTO 1993 Rigid Pavement Functions
# ═══════════════════════════════════════════════════════════════════════════════
 
ZR_TABLE = {
    50: -0.000, 60: -0.253, 70: -0.524, 75: -0.674, 80: -0.842,
    85: -1.037, 90: -1.282, 91: -1.341, 92: -1.405, 93: -1.476,
    94: -1.555, 95: -1.645, 96: -1.751, 97: -1.881, 98: -2.054,
    99: -2.327, 99.9: -3.090,
}
 
def get_ZR(r):
    levels = sorted(ZR_TABLE.keys())
    if r <= levels[0]:  return ZR_TABLE[levels[0]]
    if r >= levels[-1]: return ZR_TABLE[levels[-1]]
    for i in range(len(levels) - 1):
        if levels[i] <= r <= levels[i + 1]:
            r1, r2 = levels[i], levels[i + 1]
            return ZR_TABLE[r1] + (ZR_TABLE[r2] - ZR_TABLE[r1]) * (r - r1) / (r2 - r1)
 
 
def rigid_lhs(D, W18, ZR, S0, delta_psi, Sc, Cd, J, Ec, k):
    """
    AASHTO 1993 Rigid Pavement Design Equation (LHS - RHS = 0)
    log10(W18) = ZR*S0 + 7.35*log10(D+1) - 0.06
                 + log10[DPSI/(4.5-1.5)] / [1 + 1.624e7/(D+1)^8.46]
                 + (4.22 - 0.32*pt) * log10[ Sc*Cd*(D^0.75 - 1.132)
                       / (215.63*J*(D^0.75 - 18.42/(Ec/k)^0.25)) ]
    """
    if D <= 0:
        return float('inf')
    try:
        log_W18 = math.log10(W18)
        term1 = ZR * S0
        term2 = 7.35 * math.log10(D + 1) - 0.06
        psi_ratio = delta_psi / (4.5 - 1.5)
        if psi_ratio <= 0:
            return float('inf')
        term3 = math.log10(psi_ratio) / (1 + 1.624e7 / ((D + 1) ** 8.46))
        pt = 4.5 - delta_psi
        inner_num = Sc * Cd * (D ** 0.75 - 1.132)
        inner_den = 215.63 * J * (D ** 0.75 - 18.42 / ((Ec / k) ** 0.25))
        if inner_den <= 0 or inner_num <= 0:
            return float('inf')
        term4 = (4.22 - 0.32 * pt) * math.log10(inner_num / inner_den)
        lhs = term1 + term2 + term3 + term4
        return lhs - log_W18
    except Exception:
        return float('inf')
 
 
def solve_D(W18, ZR, S0, delta_psi, Sc, Cd, J, Ec, k):
    """Solve for slab thickness D (inches) using Brent's method"""
    try:
        # Check bounds
        f_lo = rigid_lhs(4.0, W18, ZR, S0, delta_psi, Sc, Cd, J, Ec, k)
        f_hi = rigid_lhs(30.0, W18, ZR, S0, delta_psi, Sc, Cd, J, Ec, k)
        if f_lo * f_hi > 0:
            # Try wider bounds
            for lo in [1.0, 2.0, 3.0]:
                f_lo = rigid_lhs(lo, W18, ZR, S0, delta_psi, Sc, Cd, J, Ec, k)
                if f_lo * f_hi <= 0:
                    return brentq(rigid_lhs, lo, 30.0,
                                  args=(W18, ZR, S0, delta_psi, Sc, Cd, J, Ec, k),
                                  xtol=1e-6, maxiter=500)
            return None
        return brentq(rigid_lhs, 4.0, 30.0,
                      args=(W18, ZR, S0, delta_psi, Sc, Cd, J, Ec, k),
                      xtol=1e-6, maxiter=500)
    except Exception:
        return None
 
 
def cbr_to_k(cbr):
    """CBR to modulus of subgrade reaction k (pci) — AASHTO correlation"""
    # k (pci) ≈ 10 × CBR  (approximate, for granular subgrade)
    return 10.0 * cbr
 
 
def mr_to_k(MR_psi):
    """MR (psi) to k (pci) — approximate AASHTO 1993"""
    # k ≈ MR / 19.4  (AASHTO Guide 1993, approximate)
    return MR_psi / 19.4
 
 
def Ec_from_fc(fc_mpa):
    """Modulus of elasticity from compressive strength (MPa → MPa)"""
    return 4700 * math.sqrt(fc_mpa)
 
 
def Sc_from_fc(fc_mpa):
    """Modulus of rupture (MR) from fc: fr ≈ 0.62 * sqrt(fc) MPa → psi"""
    fr_mpa = 0.62 * math.sqrt(fc_mpa)
    return fr_mpa * 145.038   # to psi
 
 
# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:14px 0 6px 0;">
        <div style="font-size:2.2rem;">🏗️</div>
        <div style="color:#7ab3d4;font-weight:700;font-size:1rem;letter-spacing:1px;">AASHTO 1993</div>
        <div style="color:#4a7a9b;font-size:0.75rem;">Rigid Pavement Design</div>
    </div>
    <hr style="border-color:#1e3a5f;margin:8px 0;">
    """, unsafe_allow_html=True)
 
    st.markdown("### ⚙️ พารามิเตอร์การออกแบบ")
 
    # ── Traffic ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>🚛 ปริมาณจราจร (W18)</h3></div>', unsafe_allow_html=True)
    w18_mode = st.radio("วิธีกรอก", ["W18 โดยตรง", "คำนวณจาก AADT"], horizontal=True)
 
    if w18_mode == "W18 โดยตรง":
        W18 = st.number_input("W18 (ESAL)", min_value=1e4, max_value=1e9,
                               value=5e6, format="%.2e", step=1e5)
    else:
        AADT        = st.number_input("AADT (คัน/วัน)", 100, 200000, 8000, 500)
        T_pct       = st.number_input("% รถบรรทุก", 1.0, 60.0, 15.0, 1.0)
        design_life = st.number_input("อายุออกแบบ (ปี)", 5, 40, 20, 5)
        TF          = st.number_input("Truck Factor (LEF)", 0.1, 5.0, 0.75, 0.05)
        growth_rate = st.number_input("อัตราการเติบโต (%/ปี)", 0.0, 10.0, 3.0, 0.5)
        if growth_rate > 0:
            GF = ((1 + growth_rate / 100) ** design_life - 1) / (growth_rate / 100)
        else:
            GF = float(design_life)
        W18 = AADT * (T_pct / 100) * 365 * GF * TF
        st.info(f"**W18 = {W18:,.0f} ESAL**")
 
    # ── Reliability ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>📊 ความน่าเชื่อถือ</h3></div>', unsafe_allow_html=True)
    R_preset = st.selectbox("ประเภททาง", [
        "ทางหลวงพิเศษ (R=99%)", "ทางหลวงระหว่างเมือง (R=95%)",
        "ทางหลวงสายหลัก (R=90%)", "ทางหลวงสายรอง (R=85%)",
        "ถนนในเมืองสายหลัก (R=95%)", "ถนนในเมืองสายรอง (R=80%)", "กำหนดเอง"])
    R_map = {
        "ทางหลวงพิเศษ (R=99%)": 99,
        "ทางหลวงระหว่างเมือง (R=95%)": 95,
        "ทางหลวงสายหลัก (R=90%)": 90,
        "ทางหลวงสายรอง (R=85%)": 85,
        "ถนนในเมืองสายหลัก (R=95%)": 95,
        "ถนนในเมืองสายรอง (R=80%)": 80,
        "กำหนดเอง": 90,
    }
    if R_preset == "กำหนดเอง":
        reliability = st.slider("Reliability (%)", 50, 99, 90)
    else:
        reliability = R_map[R_preset]
        st.info(f"R = {reliability}%")
    ZR = get_ZR(reliability)
    S0 = st.number_input("S0 (Overall Std. Dev.)", 0.30, 0.50, 0.35, 0.01,
                          help="ทางคอนกรีตแนะนำ 0.30-0.40")
 
    # ── Serviceability ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>📉 Serviceability</h3></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    pi = c1.number_input("pi (เริ่มต้น)", 3.5, 5.0, 4.5, 0.1,
                          help="ทางคอนกรีตใหม่ pi=4.5")
    pt = c2.number_input("pt (สิ้นสุด)", 1.5, 3.5, 2.5, 0.1)
    delta_psi = pi - pt
    st.info(f"ΔPSI = {delta_psi:.1f}   (pi={pi}, pt={pt})")
 
    # ── Concrete Properties ───────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>🧱 คุณสมบัติคอนกรีต</h3></div>', unsafe_allow_html=True)
    concrete_grade = st.selectbox("เกรดคอนกรีต", [
        "C30 (fc'=30 MPa)", "C35 (fc'=35 MPa)", "C40 (fc'=40 MPa)",
        "C45 (fc'=45 MPa)", "C50 (fc'=50 MPa)", "กำหนดเอง"])
    fc_map = {"C30 (fc'=30 MPa)": 30, "C35 (fc'=35 MPa)": 35,
              "C40 (fc'=40 MPa)": 40, "C45 (fc'=45 MPa)": 45,
              "C50 (fc'=50 MPa)": 50, "กำหนดเอง": 35}
    fc_mpa = fc_map[concrete_grade]
    if concrete_grade == "กำหนดเอง":
        fc_mpa = st.number_input("fc' (MPa)", 20.0, 70.0, 35.0, 1.0)
 
    Ec_mpa  = Ec_from_fc(fc_mpa)
    Ec_psi  = Ec_mpa * 145.038
    Sc_psi  = Sc_from_fc(fc_mpa)
    Sc_mpa  = Sc_psi / 145.038
 
    custom_Sc = st.checkbox("กำหนด Sc (MR) เอง", value=False)
    if custom_Sc:
        Sc_mpa = st.number_input("Sc — Modulus of Rupture (MPa)", 2.0, 8.0,
                                  round(Sc_mpa, 2), 0.05)
        Sc_psi = Sc_mpa * 145.038
 
    custom_Ec = st.checkbox("กำหนด Ec เอง", value=False)
    if custom_Ec:
        Ec_mpa = st.number_input("Ec — Elastic Modulus (MPa)", 10000, 50000,
                                  int(Ec_mpa), 500)
        Ec_psi = Ec_mpa * 145.038
 
    st.caption(f"Ec = {Ec_mpa:,.0f} MPa   |   Sc (fr) = {Sc_mpa:.2f} MPa")
 
    # ── Joint & Load Transfer ─────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>🔩 รอยต่อและการถ่ายแรง</h3></div>', unsafe_allow_html=True)
    J_opts = {
        "JPCP มี Dowel Bar (J=3.2)": 3.2,
        "JPCP ไม่มี Dowel Bar (J=3.8)": 3.8,
        "JRCP มี Tie Bar (J=3.2)": 3.2,
        "CRCP (J=2.9)": 2.9,
        "กำหนดเอง": 3.2,
    }
    J_sel = st.selectbox("ประเภทแผ่นพื้น / รอยต่อ", list(J_opts.keys()))
    J = J_opts[J_sel]
    if J_sel == "กำหนดเอง":
        J = st.number_input("Load Transfer Coefficient (J)", 2.5, 4.5, 3.2, 0.1)
    st.caption(f"J = {J}")
 
    Cd_opts = {
        "ระบายน้ำดีเยี่ยม, อิ่มตัว<1% (Cd=1.25)": 1.25,
        "ระบายน้ำดี, อิ่มตัว<5% (Cd=1.20)": 1.20,
        "ระบายน้ำปานกลาง (Cd=1.10)": 1.10,
        "ระบายน้ำแย่ (Cd=0.90)": 0.90,
        "ระบายน้ำแย่มาก (Cd=0.70)": 0.70,
        "กำหนดเอง": 1.0,
    }
    Cd_sel = st.selectbox("Drainage Coefficient (Cd)", list(Cd_opts.keys()))
    Cd = Cd_opts[Cd_sel]
    if Cd_sel == "กำหนดเอง":
        Cd = st.number_input("Cd", 0.50, 1.25, 1.0, 0.05)
    st.caption(f"Cd = {Cd}")
 
    # ── Subgrade / Foundation ─────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>🏔️ ดินพื้นทาง (Subgrade / k)</h3></div>', unsafe_allow_html=True)
    k_method = st.radio("กรอก k จาก", ["CBR (%)", "k โดยตรง (pci)", "MR (psi)"])
    if k_method == "CBR (%)":
        CBR_val = st.number_input("CBR (%)", 1.0, 30.0, 5.0, 0.5)
        k = cbr_to_k(CBR_val)
        st.info(f"k ≈ {k:.0f} pci")
    elif k_method == "k โดยตรง (pci)":
        k = float(st.number_input("k (pci)", 50, 1000, 100, 10))
        CBR_val = k / 10.0
    else:
        MR_psi = st.number_input("MR (psi)", 1000, 30000, 7500, 250)
        k = mr_to_k(MR_psi)
        CBR_val = MR_psi / 1500.0
        st.info(f"k ≈ {k:.1f} pci")
 
    # ── Base Layer ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>🪨 ชั้นรองพื้นทาง (Base/Subbase)</h3></div>', unsafe_allow_html=True)
    use_base = st.checkbox("มีชั้นรองพื้นทาง", value=True)
    if use_base:
        base_type = st.selectbox("วัสดุรองพื้น", [
            "หินคลุก / หินย่อย (Granular Base)",
            "พื้นทางซีเมนต์ (Cement Treated Base)",
            "Lean Concrete Base",
            "Asphalt Treated Base",
        ])
        base_thick_cm = st.number_input("ความหนารองพื้น (cm)", 10, 60, 20, 5)
        # Composite k adjustment (simplified — AASHTO Table 3.3)
        base_k_boost = {"หินคลุก / หินย่อย (Granular Base)": 1.2,
                        "พื้นทางซีเมนต์ (Cement Treated Base)": 1.5,
                        "Lean Concrete Base": 1.8,
                        "Asphalt Treated Base": 1.4}
        k_composite = k * base_k_boost[base_type]
        st.info(f"k composite ≈ {k_composite:.0f} pci")
    else:
        base_thick_cm = 0
        base_type = "—"
        k_composite = k
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# Solve D
# ═══════════════════════════════════════════════════════════════════════════════
D_sol = solve_D(W18, ZR, S0, delta_psi, Sc_psi, Cd, J, Ec_psi, k_composite)
if D_sol is not None:
    D_in_req  = D_sol
    D_cm_req  = D_sol * 2.54
    D_in_design = math.ceil(D_sol * 4) / 4       # round up to nearest 0.25"
    D_cm_design = D_in_design * 2.54
else:
    D_in_req = D_cm_req = D_in_design = D_cm_design = None
 
 
# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🏗️ ออกแบบผิวทางคอนกรีต — AASHTO 1993</h1>
    <p>กรมทางหลวง &nbsp;|&nbsp; Department of Highways Thailand &nbsp;|&nbsp;
       Rigid Pavement — Jointed Plain Concrete Pavement (JPCP)</p>
</div>
""", unsafe_allow_html=True)
 
tab1, tab2, tab3, tab4 = st.tabs([
    "📐 ผลการออกแบบ",
    "📊 Sensitivity Analysis",
    "📋 ตารางค่าอ้างอิง",
    "ℹ️ ทฤษฎีและสูตร",
])
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Design Results
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
 
    # ── Summary Cards ─────────────────────────────────────────────────────────
    if D_cm_design is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""<div class="result-card">
                <div class="label">ความหนาแผ่น (ออกแบบ)</div>
                <div class="value">{D_cm_design:.1f}</div>
                <div class="unit">cm ({D_in_design:.2f} in)</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="result-card">
                <div class="label">ความหนาต้องการ</div>
                <div class="value">{D_cm_req:.2f}</div>
                <div class="unit">cm ({D_in_req:.3f} in)</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="result-card-blue">
                <div class="label">Design ESAL</div>
                <div class="value">{W18:,.0f}</div>
                <div class="unit">W18</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="result-card-blue">
                <div class="label">Reliability</div>
                <div class="value">{reliability}%</div>
                <div class="unit">ZR = {ZR:.3f}</div>
            </div>""", unsafe_allow_html=True)
        with c5:
            st.markdown(f"""<div class="result-card-blue">
                <div class="label">k Composite</div>
                <div class="value">{k_composite:.0f}</div>
                <div class="unit">pci</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.error("❌ ไม่สามารถคำนวณได้ — กรุณาตรวจสอบค่าพารามิเตอร์")
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Cross Section + Properties ────────────────────────────────────────────
    col_left, col_right = st.columns([0.65, 0.35])
 
    with col_left:
        st.markdown("### 🏗️ หน้าตัดโครงสร้างทางคอนกรีต")
        if D_cm_design:
            ok = D_cm_design >= D_cm_req
            ok_color = "#4caf50" if ok else "#f44336"
            ok_label = "✅ ผ่านการออกแบบ" if ok else "❌ ความหนาไม่เพียงพอ"
 
            html = '<div style="background:#0a0f1a;border:1px solid #1e3a5f;border-radius:10px;padding:18px;">'
            html += '<div style="color:#7ab3d4;font-size:0.7rem;text-align:center;margin-bottom:12px;letter-spacing:1.5px;">RIGID PAVEMENT CROSS-SECTION</div>'
 
            # Surface treatment (optional thin AC overlay note)
            html += '<div class="layer-box" style="background:#0f1520;border:1px dashed #3a5a7a;color:#5a8aaa;font-size:0.75rem;">'
            html += '<span>⬜ Surface (Exposed Concrete / Texture)</span><span>—</span></div>'
 
            # Concrete slab
            html += f'<div class="layer-box layer-concrete">'
            html += f'<span>🔷 แผ่นคอนกรีต Portland Cement (PCC)</span>'
            html += f'<span><b>{D_cm_design:.1f} cm</b> ({D_in_design:.2f}")</span></div>'
            html += f'<div style="text-align:center;color:#3a5a7a;font-size:0.66rem;margin:-2px 0 2px 0;">'
            html += f'fc\'={fc_mpa} MPa | Ec={Ec_mpa:,.0f} MPa | Sc={Sc_mpa:.2f} MPa | J={J} | Cd={Cd}</div>'
 
            if use_base:
                html += f'<div class="layer-box layer-base">'
                html += f'<span>🟢 {base_type}</span>'
                html += f'<span><b>{base_thick_cm} cm</b></span></div>'
 
            html += f'<div class="layer-box layer-subgrade">'
            html += f'<span>🟤 Subgrade</span><span>k = {k:.0f} pci | CBR ≈ {CBR_val:.1f}%</span></div>'
 
            total_thickness = D_cm_design + base_thick_cm
            html += f'<hr style="border-color:#1e3a5f;margin:10px 0;">'
            html += f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#7ab3d4;">'
            html += f'<span>รวม <b style="color:#c8f5c8">{total_thickness:.1f} cm</b></span>'
            html += f'<span style="color:{ok_color};font-weight:700;">{ok_label}</span></div>'
            html += f'<div style="text-align:center;color:#4a7a9b;font-size:0.72rem;margin-top:4px;">'
            html += f'D required = {D_cm_req:.2f} cm | D design = {D_cm_design:.1f} cm</div>'
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)
 
        # Summary Table
        st.markdown("### 📊 สรุปผลการออกแบบ")
        if D_cm_design:
            rows = [
                {"พารามิเตอร์": "W18 (ESAL)", "ค่า": f"{W18:,.0f}", "หน่วย": "—"},
                {"พารามิเตอร์": "Reliability (R)", "ค่า": f"{reliability}%", "หน่วย": "—"},
                {"พารามิเตอร์": "ZR", "ค่า": f"{ZR:.4f}", "หน่วย": "—"},
                {"พารามิเตอร์": "S0", "ค่า": f"{S0:.2f}", "หน่วย": "—"},
                {"พารามิเตอร์": "ΔPSI", "ค่า": f"{delta_psi:.1f}", "หน่วย": "—"},
                {"พารามิเตอร์": "fc' คอนกรีต", "ค่า": f"{fc_mpa}", "หน่วย": "MPa"},
                {"พารามิเตอร์": "Ec", "ค่า": f"{Ec_mpa:,.0f}", "หน่วย": "MPa"},
                {"พารามิเตอร์": "Sc (Modulus of Rupture)", "ค่า": f"{Sc_mpa:.3f}", "หน่วย": "MPa"},
                {"พารามิเตอร์": "Load Transfer J", "ค่า": f"{J}", "หน่วย": "—"},
                {"พารามิเตอร์": "Drainage Coefficient Cd", "ค่า": f"{Cd}", "หน่วย": "—"},
                {"พารามิเตอร์": "k Subgrade", "ค่า": f"{k:.1f}", "หน่วย": "pci"},
                {"พารามิเตอร์": "k Composite", "ค่า": f"{k_composite:.1f}", "หน่วย": "pci"},
                {"พารามิเตอร์": "D ที่ต้องการ", "ค่า": f"{D_cm_req:.2f}", "หน่วย": "cm"},
                {"พารามิเตอร์": "D ออกแบบ (round-up 0.25\")", "ค่า": f"{D_cm_design:.1f}", "หน่วย": "cm"},
                {"พารามิเตอร์": "ชั้นรองพื้น", "ค่า": base_type if use_base else "—", "หน่วย": f"{base_thick_cm} cm" if use_base else "—"},
                {"พารามิเตอร์": "ความหนารวม", "ค่า": f"{D_cm_design + base_thick_cm:.1f}", "หน่วย": "cm"},
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
 
            if D_cm_design >= D_cm_req:
                st.success(f"✅ ผ่านการออกแบบ — D = {D_cm_design:.1f} cm >= D ที่ต้องการ {D_cm_req:.2f} cm")
            else:
                st.error(f"❌ ไม่ผ่าน — D = {D_cm_design:.1f} cm < D ที่ต้องการ {D_cm_req:.2f} cm")
 
    with col_right:
        st.markdown("### 📌 คุณสมบัติวัสดุ")
        st.markdown(f"""
        <div class="info-box">
        <b style="color:#a8d0f0;">คอนกรีต:</b><br>
        &bull; เกรด: {concrete_grade}<br>
        &bull; fc' = {fc_mpa} MPa<br>
        &bull; Ec = {Ec_mpa:,.0f} MPa ({Ec_psi:,.0f} psi)<br>
        &bull; Sc (fr) = {Sc_mpa:.3f} MPa ({Sc_psi:.1f} psi)<br><br>
        <b style="color:#a8d0f0;">รอยต่อ:</b><br>
        &bull; J = {J} ({J_sel.split('(')[0].strip()})<br>
        &bull; Cd = {Cd}<br><br>
        <b style="color:#a8d0f0;">ดินพื้นทาง:</b><br>
        &bull; k = {k:.0f} pci<br>
        &bull; k composite = {k_composite:.0f} pci<br>
        &bull; CBR ≈ {CBR_val:.1f}%<br><br>
        <b style="color:#a8d0f0;">การออกแบบ:</b><br>
        &bull; D required = {D_cm_req:.2f} cm<br>
        &bull; D design = {D_cm_design:.1f} cm<br>
        &bull; D design = {D_in_design:.2f} in
        </div>
        """, unsafe_allow_html=True) if D_cm_design else st.error("คำนวณไม่ได้")
 
        # Bar chart: concrete vs base vs subgrade
        if D_cm_design:
            names  = ["แผ่น PCC", "ชั้นรอง", "ความหนารวม"]
            values = [D_cm_design, base_thick_cm if use_base else 0, D_cm_design + (base_thick_cm if use_base else 0)]
            colors = ["#4a6fa8", "#4a7a4a", "#8a5a8a"]
            fig_b = go.Figure(go.Bar(
                x=names, y=values,
                marker_color=colors,
                text=[f"{v:.1f} cm" for v in values],
                textposition="outside",
            ))
            fig_b.update_layout(
                title="ความหนาโครงสร้าง",
                yaxis_title="ความหนา (cm)",
                template="plotly_dark",
                paper_bgcolor="#0a0f1a", plot_bgcolor="#0a0f1a",
                font=dict(family="Sarabun", color="#7ab3d4", size=10),
                height=260, margin=dict(l=10, r=20, t=40, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_b, use_container_width=True)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Sensitivity Analysis — ผิวทางคอนกรีต AASHTO 1993")
 
    if D_cm_design is None:
        st.warning("ไม่สามารถทำ Sensitivity Analysis ได้ — กรุณาตรวจสอบค่าพารามิเตอร์")
    else:
        c1, c2 = st.columns(2)
 
        # D vs W18
        with c1:
            W18r = np.logspace(4, 9, 80)
            Dr   = [solve_D(w, ZR, S0, delta_psi, Sc_psi, Cd, J, Ec_psi, k_composite) for w in W18r]
            Dr_cm = [d * 2.54 if d else np.nan for d in Dr]
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=W18r, y=Dr_cm, mode='lines',
                                      line=dict(color='#0d74e7', width=2.5),
                                      name="D required"))
            fig1.add_vline(x=W18, line_dash="dash", line_color="#f0a030",
                           annotation_text=f"W18={W18:.1e}", annotation_font_color="#f0a030")
            fig1.add_hline(y=D_cm_design, line_dash="dash", line_color="#4caf50",
                           annotation_text=f"D={D_cm_design:.1f}cm", annotation_font_color="#4caf50")
            fig1.update_layout(title="D vs W18 (ESAL)", xaxis_title="W18 (ESAL)",
                               yaxis_title="D ที่ต้องการ (cm)", xaxis_type="log",
                               template="plotly_dark",
                               paper_bgcolor="#0d1117", plot_bgcolor="#111d2e",
                               font=dict(family="Sarabun", color="#7ab3d4"), height=340)
            st.plotly_chart(fig1, use_container_width=True)
 
        # D vs k
        with c2:
            kr   = np.linspace(30, 600, 80)
            Dkr  = [solve_D(W18, ZR, S0, delta_psi, Sc_psi, Cd, J, Ec_psi, kk) for kk in kr]
            Dkr_cm = [d * 2.54 if d else np.nan for d in Dkr]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=kr, y=Dkr_cm, mode='lines',
                                      line=dict(color='#e74c3c', width=2.5)))
            fig2.add_vline(x=k_composite, line_dash="dash", line_color="#f0a030",
                           annotation_text=f"k={k_composite:.0f}", annotation_font_color="#f0a030")
            fig2.update_layout(title="D vs k (Subgrade Reaction)",
                               xaxis_title="k (pci)", yaxis_title="D ที่ต้องการ (cm)",
                               template="plotly_dark",
                               paper_bgcolor="#0d1117", plot_bgcolor="#111d2e",
                               font=dict(family="Sarabun", color="#7ab3d4"), height=340)
            st.plotly_chart(fig2, use_container_width=True)
 
        c3, c4 = st.columns(2)
 
        # D vs Sc (MR)
        with c3:
            Sc_r   = np.linspace(300, 900, 80)
            DSc_cm = [solve_D(W18, ZR, S0, delta_psi, sc, Cd, J, Ec_psi, k_composite) for sc in Sc_r]
            DSc_cm2 = [d * 2.54 if d else np.nan for d in DSc_cm]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=Sc_r, y=DSc_cm2, mode='lines',
                                      line=dict(color='#9b59b6', width=2.5)))
            fig3.add_vline(x=Sc_psi, line_dash="dash", line_color="#f0a030",
                           annotation_text=f"Sc={Sc_psi:.0f} psi", annotation_font_color="#f0a030")
            fig3.update_layout(title="D vs Sc (Modulus of Rupture)",
                               xaxis_title="Sc (psi)", yaxis_title="D ที่ต้องการ (cm)",
                               template="plotly_dark",
                               paper_bgcolor="#0d1117", plot_bgcolor="#111d2e",
                               font=dict(family="Sarabun", color="#7ab3d4"), height=340)
            st.plotly_chart(fig3, use_container_width=True)
 
        # D vs Reliability
        with c4:
            Rr   = [50, 60, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
            DRr  = [solve_D(W18, get_ZR(r), S0, delta_psi, Sc_psi, Cd, J, Ec_psi, k_composite) for r in Rr]
            DRr_cm = [d * 2.54 if d else np.nan for d in DRr]
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=Rr, y=DRr_cm, mode='lines+markers',
                                      line=dict(color='#1abc9c', width=2.5), marker=dict(size=5)))
            fig4.add_vline(x=reliability, line_dash="dash", line_color="#f0a030",
                           annotation_text=f"R={reliability}%", annotation_font_color="#f0a030")
            fig4.update_layout(title="D vs Reliability (%)",
                               xaxis_title="R (%)", yaxis_title="D ที่ต้องการ (cm)",
                               template="plotly_dark",
                               paper_bgcolor="#0d1117", plot_bgcolor="#111d2e",
                               font=dict(family="Sarabun", color="#7ab3d4"), height=340)
            st.plotly_chart(fig4, use_container_width=True)
 
        # D vs J (Load Transfer)
        st.markdown("#### ผลของ Load Transfer Coefficient (J) ต่อความหนาแผ่น")
        Jr   = np.linspace(2.5, 4.5, 50)
        DJr  = [solve_D(W18, ZR, S0, delta_psi, Sc_psi, Cd, jj, Ec_psi, k_composite) for jj in Jr]
        DJr_cm = [d * 2.54 if d else np.nan for d in DJr]
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=Jr, y=DJr_cm, mode='lines',
                                  line=dict(color='#f39c12', width=2.5)))
        fig5.add_vline(x=J, line_dash="dash", line_color="#4caf50",
                       annotation_text=f"J={J}", annotation_font_color="#4caf50")
        fig5.update_layout(title="D vs Load Transfer Coefficient (J)",
                           xaxis_title="J", yaxis_title="D ที่ต้องการ (cm)",
                           template="plotly_dark",
                           paper_bgcolor="#0d1117", plot_bgcolor="#111d2e",
                           font=dict(family="Sarabun", color="#7ab3d4"), height=300)
        st.plotly_chart(fig5, use_container_width=True)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Reference Tables
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 ตารางค่าอ้างอิง — AASHTO 1993 Rigid Pavement")
 
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Load Transfer Coefficient (J)")
        st.dataframe(pd.DataFrame({
            "ประเภทแผ่น": [
                "JPCP มี Dowel Bar", "JPCP ไม่มี Dowel Bar",
                "JRCP มี Tie Bar", "CRCP",
            ],
            "ไหล่ทาง Asphalt": [3.2, 3.8, 3.2, 2.9],
            "ไหล่ทาง Concrete": [2.7, 3.2, 2.7, 2.3],
        }), use_container_width=True, hide_index=True)
 
        st.markdown("#### Drainage Coefficient (Cd)")
        st.dataframe(pd.DataFrame({
            "คุณภาพระบายน้ำ": ["ดีเยี่ยม (<2 hr)", "ดี (2-24 hr)",
                                "ปานกลาง (1-7 วัน)", "แย่ (>1 เดือน)"],
            "อิ่มตัว <1%":  [1.25, 1.20, 1.15, 1.05],
            "อิ่มตัว 1-5%": [1.20, 1.15, 1.10, 1.00],
            "อิ่มตัว 5-25%":[1.15, 1.10, 1.00, 0.90],
            "อิ่มตัว >25%": [1.10, 1.00, 0.90, 0.70],
        }), use_container_width=True, hide_index=True)
 
        st.markdown("#### ZR — Standard Normal Deviate")
        st.dataframe(pd.DataFrame({
            "R (%)": [50, 60, 70, 75, 80, 85, 90, 95, 98, 99],
            "ZR":    [ZR_TABLE[r] for r in [50, 60, 70, 75, 80, 85, 90, 95, 98, 99]],
        }), use_container_width=True, hide_index=True)
 
    with c2:
        st.markdown("#### คุณสมบัติคอนกรีตตามเกรด")
        grades = [30, 35, 40, 45, 50]
        st.dataframe(pd.DataFrame({
            "เกรด": [f"C{g}" for g in grades],
            "fc' (MPa)": grades,
            "Ec (MPa)": [round(Ec_from_fc(g)) for g in grades],
            "Sc/fr (MPa)": [round(0.62 * math.sqrt(g), 3) for g in grades],
            "Sc (psi)": [round(0.62 * math.sqrt(g) * 145.038, 1) for g in grades],
        }), use_container_width=True, hide_index=True)
 
        st.markdown("#### k จาก CBR (ประมาณ)")
        cbr_vals = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]
        st.dataframe(pd.DataFrame({
            "CBR (%)": cbr_vals,
            "k (pci)": [cbr_to_k(c) for c in cbr_vals],
            "MR (psi)": [c * 1500 for c in cbr_vals],
            "MR (MPa)": [round(c * 1500 / 145.038, 1) for c in cbr_vals],
        }), use_container_width=True, hide_index=True)
 
        st.markdown("#### D ออกแบบแนะนำ (กรมทางหลวง)")
        st.dataframe(pd.DataFrame({
            "ประเภทถนน": ["ทางหลวงพิเศษ", "ทางหลวงระหว่างเมือง",
                          "ทางหลวงสายหลัก", "ถนนในเมือง", "ถนนสายรอง"],
            "D แนะนำ (cm)": ["28-32", "25-30", "22-28", "20-26", "18-22"],
            "fc' แนะนำ (MPa)": [40, 35, 35, 30, 30],
        }), use_container_width=True, hide_index=True)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Theory
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ℹ️ ทฤษฎีและสูตร AASHTO 1993 — ผิวทางคอนกรีต")
 
    st.markdown("""
    <div class="formula-box">
    <b style="color:#a8d0f0;">AASHTO 1993 Rigid Pavement Design Equation:</b><br><br>
    log(W18) = ZR × S0 + 7.35 × log(D+1) − 0.06<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ log[ΔPSI / (4.5−1.5)] / [1 + 1.624×10⁷ / (D+1)⁸·⁴⁶]<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ (4.22 − 0.32×pt) × log[ Sc×Cd×(D⁰·⁷⁵ − 1.132) / (215.63×J×(D⁰·⁷⁵ − 18.42/(Ec/k)⁰·²⁵)) ]<br><br>
    <b style="color:#a8d0f0;">ตัวแปร:</b><br>
    &nbsp; W18  = ESAL (จำนวนแกนมาตรฐาน 18-kip)<br>
    &nbsp; ZR   = Standard Normal Deviate (จาก Reliability R)<br>
    &nbsp; S0   = Overall Standard Deviation (ทางคอนกรีต: 0.30-0.40)<br>
    &nbsp; D    = ความหนาแผ่นคอนกรีต (นิ้ว)<br>
    &nbsp; ΔPSI = pi − pt (Serviceability Loss)<br>
    &nbsp; pt   = Terminal Serviceability Index (2.0-2.5)<br>
    &nbsp; Sc   = Modulus of Rupture (psi): fr ≈ 0.62√fc' (MPa)<br>
    &nbsp; Cd   = Drainage Coefficient (0.70-1.25)<br>
    &nbsp; J    = Load Transfer Coefficient (2.5-4.5)<br>
    &nbsp; Ec   = Elastic Modulus of Concrete (psi): ≈ 4700√fc' (MPa)<br>
    &nbsp; k    = Modulus of Subgrade Reaction (pci)<br><br>
    <b style="color:#a8d0f0;">ค่าแปลง:</b><br>
    &nbsp; 1 MPa = 145.038 psi &nbsp;|&nbsp; 1 in = 2.54 cm<br>
    &nbsp; k (pci) ≈ 10 × CBR (%)  &nbsp;|&nbsp; MR (psi) ≈ 1,500 × CBR (%)
    </div>
    """, unsafe_allow_html=True)
 
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-box">
        <b style="color:#a8d0f0;">ขั้นตอนออกแบบ AASHTO 1993 Rigid:</b><br>
        1. กำหนด W18 จากการสำรวจและพยากรณ์จราจร<br>
        2. เลือก Reliability (R) ตามประเภทถนน<br>
        3. กำหนด S0 = 0.35 (ค่าแนะนำทางคอนกรีต)<br>
        4. กำหนด pi=4.5, pt=2.5 → ΔPSI=2.0<br>
        5. กำหนดเกรดคอนกรีต → Ec, Sc<br>
        6. กำหนด J (ประเภทรอยต่อ) และ Cd (ระบายน้ำ)<br>
        7. วัด k ดินพื้นทาง (หรือแปลงจาก CBR)<br>
        8. ปรับ k composite (ถ้ามีชั้นรอง)<br>
        9. แก้สมการหา D (นิ้ว) → ปัดขึ้น 0.25"<br>
        10. ตรวจสอบกับ D ขั้นต่ำของกรมทางหลวง
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-box">
        <b style="color:#a8d0f0;">ประเภทแผ่นคอนกรีต (Pavement Types):</b><br>
        🔷 <b>JPCP</b> — Jointed Plain Concrete Pavement<br>
        &nbsp;&nbsp;&nbsp; รอยต่อทุก 4-6 เมตร มี/ไม่มี Dowel Bar<br>
        🔷 <b>JRCP</b> — Jointed Reinforced Concrete Pavement<br>
        &nbsp;&nbsp;&nbsp; รอยต่อทุก 8-12 เมตร มีเหล็กเสริม<br>
        🔷 <b>CRCP</b> — Continuously Reinforced Concrete Pavement<br>
        &nbsp;&nbsp;&nbsp; ไม่มีรอยต่อตามขวาง เหล็กเสริมต่อเนื่อง<br><br>
        <b style="color:#a8d0f0;">อ้างอิง:</b><br>
        AASHTO (1993). Guide for Design of Pavement Structures.<br>
        กรมทางหลวง (2566). มาตรฐานผิวทางคอนกรีต.<br>
        Huang, Y.H. (2004). Pavement Analysis and Design, 2nd Ed.
        </div>
        """, unsafe_allow_html=True)
 
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#1e3a5f;margin:32px 0 12px 0;">
<div style="text-align:center;color:#2d5a8a;font-size:0.78rem;font-family:'Sarabun',sans-serif;">
    🏗️ กรมทางหลวง — Department of Highways Thailand |
    AASHTO 1993 Rigid Pavement Design — Concrete Pavement<br>
    <span style="font-size:0.68rem;color:#1e3a5f;">
        สร้างด้วย Streamlit + Python | สำหรับการศึกษาและประกอบการออกแบบเบื้องต้นเท่านั้น
    </span>
</div>
""", unsafe_allow_html=True)
 
