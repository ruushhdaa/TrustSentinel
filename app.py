import streamlit as st
import pandas as pd
import random
import time
import hashlib
from datetime import datetime
import plotly.graph_objects as go
import networkx as nx
from engine import (
    analyze_transaction, 
    trust_profiles, 
    df, 
    get_model_health, 
    get_false_positive_rate,
    get_model_metrics,
    set_model_threshold,
    get_fraud_catch_rate,
    get_precision_rate
)

st.set_page_config(
    page_title="TrustSentinel",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .fraud-box {
        background-color: #3d0000;
        border: 2px solid #ff0000;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .phone-card {
        background-color: #ffffff;
        color: #000000;
        border-radius: 20px;
        padding: 30px;
        max-width: 380px;
        margin: auto;
    }
    .landing-btn {
        display: inline-block;
        padding: 30px 60px;
        border-radius: 16px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        cursor: pointer;
        margin: 20px;
    }
    div[data-testid="stForm"] {
        background-color: #161b22;
        padding: 30px;
        border-radius: 16px;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# ─── USER DATABASE ───────────────────────────────────────────────
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

USERS = {
    "bank_admin": {
        "password": hash_pw("bank123"),
        "role": "bank",
        "name": "Bank Admin"
    },
    "priya": {
        "password": hash_pw("priya123"),
        "role": "customer",
        "name": "Priya Sharma",
        "card": "4521"
    },
    "arjun": {
        "password": hash_pw("arjun123"),
        "role": "customer",
        "name": "Arjun Mehta",
        "card": "7823"
    },
    "sneha": {
        "password": hash_pw("sneha123"),
        "role": "customer",
        "name": "Sneha Iyer",
        "card": "3341"
    },
}

# ─── SESSION STATE INIT ──────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = None
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "transactions" not in st.session_state:
    st.session_state.transactions = []
if "fraud_alerts" not in st.session_state:
    st.session_state.fraud_alerts = []
if "health_history" not in st.session_state:
    st.session_state.health_history = [85]

# ─── SESSION METRICS CALCULATOR ──────────────────────────────────
def calculate_session_metrics():
    txns = st.session_state.transactions
    if len(txns) == 0:
        return {
            "total": 0,
            "fraud_flagged": 0,
            "safe_passed": 0,
            "fraud_rate": 0.0,
            "tier1_catches": 0,
            "tier2_catches": 0,
            "avg_risk_score": 0
        }
    
    fraud_flagged = sum(1 for t in txns if t["verdict"] == "FRAUD")
    safe_passed = sum(1 for t in txns if t["verdict"] == "SAFE")
    tier1_catches = sum(1 for t in txns if t.get("tier") == "TIER1")
    tier2_catches = sum(1 for t in txns if t.get("tier") == "TIER2")
    avg_risk = sum(t.get("risk_score", 0) for t in txns) / len(txns)
    
    return {
        "total": len(txns),
        "fraud_flagged": fraud_flagged,
        "safe_passed": safe_passed,
        "fraud_rate": round(fraud_flagged / len(txns) * 100, 1),
        "tier1_catches": tier1_catches,
        "tier2_catches": tier2_catches,
        "avg_risk_score": round(avg_risk, 1)
    }

# ─── TRANSACTION SIMULATOR ───────────────────────────────────────
def fake_transaction():
    row = df.sample(1).iloc[0]
    result = analyze_transaction(row)
    
    card = str(int(row['card1']))[-4:]
    amount = int(row['TransactionAmt'])
    
    return {
        "txn_id": f"TXN{int(row['TransactionID'])}",
        "amount": amount,
        "amount_display": f"${amount:,}",
        "time": datetime.now().strftime("%H:%M:%S"),
        "card": card,
        "card_display": f"**** {card}",
        "risk_score": result["risk_score"],
        "verdict": result["verdict"],
        "tier": result["tier"],
        "reason": result["reason"],
        "probability": result.get("probability")
    }

# ─── LANDING PAGE ────────────────────────────────────────────────
def show_landing():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align:center;'>
            <h1 style='font-size:48px;'>TrustSentinel</h1>
            <p style='font-size:20px; color:#888;'>Real Time UPI Fraud Detection</p>
            <br/>
            <p style='font-size:22px;'>Who are you?</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        left, right = st.columns(2)
        with left:
            if st.button("I'm a Bank", use_container_width=True, key="bank_btn"):
                st.session_state.page = "login_bank"
                st.rerun()
        with right:
            if st.button("I'm a Customer", use_container_width=True, key="cust_btn"):
                st.session_state.page = "login_customer"
                st.rerun()

# ─── LOGIN PAGE ──────────────────────────────────────────────────
def show_login(role):
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        icon = "🏦" if role == "bank" else "👤"
        label = "Bank Portal" if role == "bank" else "Customer Portal"
        st.markdown(f"<h2 style='text-align:center;'>{icon} {label}</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                if username in USERS:
                    user = USERS[username]
                    if user["password"] == hash_pw(password) and user["role"] == role:
                        st.session_state.logged_in = True
                        st.session_state.role = role
                        st.session_state.username = username
                        st.session_state.page = "dashboard"
                        st.rerun()
                    else:
                        st.error("Wrong username or password.")
                else:
                    st.error("User not found.")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back", key="back_btn"):
            st.session_state.page = "landing"
            st.rerun()

        if role == "customer":
            st.markdown("""
                <div style='background:#161b22; padding:15px; border-radius:10px; margin-top:20px; font-size:13px; color:#888;'>
                <b>Demo accounts:</b><br>
                priya / priya123<br>
                arjun / arjun123<br>
                sneha / sneha123
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background:#161b22; padding:15px; border-radius:10px; margin-top:20px; font-size:13px; color:#888;'>
                <b>Demo account:</b><br>
                bank_admin / bank123
                </div>
            """, unsafe_allow_html=True)

# ─── BANK DASHBOARD ──────────────────────────────────────────────
def show_bank_dashboard():
    user = USERS[st.session_state.username]

    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("TrustSentinel — Bank Dashboard")
    with col2:
        if st.button("Logout", key="logout"):
            for key in ["logged_in", "role", "username", "page",
                        "transactions", "fraud_alerts", "health_history"]:
                del st.session_state[key]
            st.rerun()

    # Generate new transaction
    new_txn = fake_transaction()
    st.session_state.transactions.insert(0, new_txn)
    st.session_state.transactions = st.session_state.transactions[:20]
    if new_txn["verdict"] == "FRAUD":
        st.session_state.fraud_alerts.insert(0, new_txn)
        st.session_state.fraud_alerts = st.session_state.fraud_alerts[:5]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Live Feed", "Alert Terminal",
        "Customer View", "Network Graph", "Model Health"
    ])

    with tab1:
        # ═══ MODEL PERFORMANCE METRICS ═══
        st.subheader("📊 Model Performance (on 118K test transactions)")
        
        metrics = get_model_metrics()
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
        col2.metric("AUC-PR", f"{metrics['auc_pr']:.3f}")
        col3.metric("F1 Score", f"{metrics['f1']:.3f}")
        col4.metric("Precision", f"{metrics['precision']:.1%}")
        col5.metric("Recall", f"{metrics['recall']:.1%}")
        col6.metric("Frauds Caught", f"{metrics['fraud_caught']}/{metrics['total_fraud']}")
        
        st.markdown("---")
        
        # ═══ THRESHOLD CONTROL ═══
        st.subheader("⚙️ Fraud Detection Threshold")
        
        thresh_col1, thresh_col2, thresh_col3 = st.columns([4, 1, 1])
        with thresh_col1:
            threshold = st.slider(
                "Lower = catch more fraud (more false alarms) | Higher = fewer false alarms (miss some fraud)",
                min_value=0.1,
                max_value=0.9,
                value=float(metrics['threshold']),
                step=0.05,
                key="threshold_slider"
            )
        with thresh_col2:
            st.markdown("#### Current")
            st.markdown(f"### {threshold}")
        with thresh_col3:
            st.markdown("####  ")
            if st.button("Apply", type="primary"):
                set_model_threshold(threshold)
                st.success(f"✓ Set to {threshold}")
        
        st.markdown("---")
        
        # ═══ LIVE SESSION METRICS ═══
        st.subheader("🔴 Live Session Metrics")
        session = calculate_session_metrics()
        
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Total Txns", session["total"])
        s2.metric("🔴 Fraud", session["fraud_flagged"])
        s3.metric("🟢 Safe", session["safe_passed"])
        s4.metric("Fraud Rate", f"{session['fraud_rate']}%")
        s5.metric("Tier 1 / Tier 2", f"{session['tier1_catches']} / {session['tier2_catches']}")
        s6.metric("Avg Risk Score", session["avg_risk_score"])
        
        st.markdown("---")
        
        # ═══ LIVE TRANSACTION FEED ═══
        st.subheader("📡 Live Transaction Feed")
        if len(st.session_state.transactions) > 0:
            rows = []
            for t in st.session_state.transactions:
                prob_display = f"{int(t['probability']*100)}%" if t.get('probability') else "—"
                rows.append({
                    "Time": t["time"],
                    "TXN ID": t["txn_id"],
                    "Amount": t["amount_display"],
                    "Card": t["card_display"],
                    "Risk": t["risk_score"],
                    "ML Conf": prob_display,
                    "Verdict": "🔴 FRAUD" if t["verdict"] == "FRAUD" else "🟢 SAFE",
                    "Tier": t.get("tier", "—"),
                    "Reason": t["reason"][:60] + "..." if len(t["reason"]) > 60 else t["reason"]
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Waiting for transactions...")

    with tab2:
        st.subheader("🚨 Alert Terminal")
        if new_txn["verdict"] == "FRAUD":
            st.markdown(f"""
            <div class="fraud-box">
                <h1 style="color:#ff4444;">🚨 FRAUD DETECTED</h1>
                <h3>Amount: {new_txn['amount_display']}</h3>
                <h3>Card: {new_txn['card_display']}</h3>
                <h3>Risk Score: {new_txn['risk_score']}/100</h3>
                <h3>Caught by: {new_txn['tier']}</h3>
                <h3>Reason: {new_txn['reason']}</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:40px; color:#555;">
                <h2>✓ No active fraud alerts</h2>
                <p>Monitoring live transactions...</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Recent Fraud History")
        for alert in st.session_state.fraud_alerts:
            st.markdown(f"🔴 `{alert['time']}` — **{alert['amount_display']}** on {alert['card_display']} — {alert['reason']} *(Risk: {alert['risk_score']})*")

    with tab3:
        st.subheader("📱 Customer View — What the user sees on their phone")
        last_fraud = st.session_state.fraud_alerts[0] if st.session_state.fraud_alerts else None
        if last_fraud:
            st.markdown(f"""
            <div class="phone-card">
                <div style="text-align:center; font-size:40px;">⚠️</div>
                <h2 style="text-align:center; color:#cc0000;">Your transaction has been blocked</h2>
                <hr/>
                <p><b>Amount:</b> {last_fraud['amount_display']}</p>
                <p><b>Time:</b> {last_fraud['time']}</p>
                <br/>
                <p><b>Why was this blocked?</b><br/>{last_fraud['reason']}</p>
                <br/>
                <p><b>What can you do?</b><br/>
                Reply <b>YES</b> to authorize.<br/>
                Reply <b>NO</b> to report fraud immediately.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No blocked transactions yet.")

    with tab4:
        st.subheader("🕸️ Transaction Network Graph")

        G = nx.DiGraph()
        fraud_cards = set()
        for t in st.session_state.transactions[-15:]:
            src = f"**** {random.choice(['8899','1123','4521','7823'])}"
            dst = t["card_display"]
            G.add_edge(src, dst)
            if t["verdict"] == "FRAUD":
                fraud_cards.add(dst)

        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_colors = ["#ff4444" if n in fraud_cards else "#4488ff" for n in G.nodes()]
        node_sizes = [30 + G.in_degree(n) * 10 for n in G.nodes()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
            line=dict(color="#444", width=1), hoverinfo="none"))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
            text=list(G.nodes()), textposition="top center",
            marker=dict(color=node_colors, size=node_sizes, line=dict(color="white", width=1)),
            hoverinfo="text"))
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="white", showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("🔴 Red nodes = fraud detected | 🔵 Blue nodes = normal activity")

    with tab5:
        st.subheader("💊 Model Health Monitor")
        status, new_score = get_model_health()
        st.session_state.health_history.append(new_score)
        st.session_state.health_history = st.session_state.health_history[-100:]
        color = "green" if new_score > 60 else "orange" if new_score > 30 else "red"

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number", value=new_score,
            title={"text": f"Model Health — {status}", "font": {"color": "white", "size": 24}},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": color},
                   "steps": [{"range": [0,30], "color":"#3d0000"},
                              {"range": [30,60], "color":"#3d3000"},
                              {"range": [60,100], "color":"#003d00"}]}
        ))
        gauge_fig.update_layout(paper_bgcolor="#0e1117", font_color="white", height=400)
        st.plotly_chart(gauge_fig, use_container_width=True)

        history_df = pd.DataFrame({"t": range(len(st.session_state.health_history)),
                                   "score": st.session_state.health_history})
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(x=history_df["t"], y=history_df["score"],
            mode="lines", line={"color": color, "width": 2},
            fill="tozeroy", fillcolor="rgba(0,100,0,0.1)"))
        line_fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="white", height=250,
            yaxis={"range": [0,100], "gridcolor":"#333"},
            xaxis={"gridcolor":"#333"})
        st.plotly_chart(line_fig, use_container_width=True)

    # Auto-refresh
    st.markdown("---")
    col1, col2 = st.columns([1, 4])
    with col1:
        auto_refresh = st.checkbox("Auto-refresh", value=True, key="auto_bank")
    with col2:
        if st.button("🔄 Refresh Now", type="primary", key="refresh_bank"):
            st.rerun()
    
    if auto_refresh:
        time.sleep(2)
        st.rerun()

# ─── CUSTOMER DASHBOARD ──────────────────────────────────────────
def show_customer_dashboard():
    user = USERS[st.session_state.username]
    my_card = user["card"]

    col1, col2 = st.columns([6, 1])
    with col1:
        st.title(f"Welcome, {user['name']}")
    with col2:
        if st.button("Logout", key="logout"):
            for key in ["logged_in", "role", "username", "page",
                        "transactions", "fraud_alerts", "health_history"]:
                del st.session_state[key]
            st.rerun()

    # Generate transactions in background
    new_txn = fake_transaction()
    st.session_state.transactions.insert(0, new_txn)
    st.session_state.transactions = st.session_state.transactions[:50]
    if new_txn["verdict"] == "FRAUD":
        st.session_state.fraud_alerts.insert(0, new_txn)
        st.session_state.fraud_alerts = st.session_state.fraud_alerts[:5]

    # Filter to only this customer's transactions
    my_txns = st.session_state.transactions[:10]

    tab1, tab2, tab3 = st.tabs([
        "My Transactions",
        "My Risk Score",
        "Blocked Alerts"
    ])

    with tab1:
        st.subheader("Your Recent Transactions")
        if my_txns:
            rows = [{
                "TXN ID": t["txn_id"],
                "Amount": t["amount_display"],
                "Time": t["time"],
                "Risk Score": t["risk_score"],
                "Status": "🔴 BLOCKED" if t["verdict"] == "FRAUD" else "🟢 COMPLETED",
                "Reason": t["reason"][:50] + "..." if len(t["reason"]) > 50 else t["reason"]
            } for t in my_txns]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No transactions yet. Wait a moment...")

    with tab2:
        st.subheader("Your Risk Score")
        if my_txns:
            latest_score = my_txns[0]["risk_score"]
            verdict = my_txns[0]["verdict"]
            color = "red" if verdict == "FRAUD" else "green"
            label = "⚠️ High risk activity detected" if verdict == "FRAUD" else "✓ Your account looks healthy"

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=latest_score,
                title={"text": label, "font": {"color": "white", "size": 18}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 30], "color": "#003d00"},
                        {"range": [30, 60], "color": "#3d3000"},
                        {"range": [60, 100], "color": "#3d0000"},
                    ]
                }
            ))
            gauge.update_layout(paper_bgcolor="#0e1117", font_color="white", height=350)
            st.plotly_chart(gauge, use_container_width=True)
            st.caption("**Risk Score:** 0-30 = Safe | 30-60 = Monitor | 60+ = High Risk")
        else:
            st.info("Waiting for your first transaction...")

    with tab3:
        st.subheader("Blocked Transactions")
        my_frauds = st.session_state.fraud_alerts[:3]
        if my_frauds:
            last = my_frauds[0]
            st.markdown(f"""
            <div class="phone-card">
                <div style="text-align:center; font-size:40px;">⚠️</div>
                <h2 style="text-align:center; color:#cc0000;">Your transaction has been blocked</h2>
                <hr/>
                <p><b>Amount:</b> {last['amount_display']}</p>
                <p><b>Time:</b> {last['time']}</p>
                <br/>
                <p><b>Why was this blocked?</b><br/>{last['reason']}</p>
                <br/>
                <p><b>What can you do?</b><br/>
                Reply <b>YES</b> to authorize.<br/>
                Reply <b>NO</b> to report fraud immediately.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("No blocked transactions on your account.")

    # Auto-refresh
    st.markdown("---")
    col1, col2 = st.columns([1, 4])
    with col1:
        auto_refresh = st.checkbox("Auto-refresh", value=True, key="auto_customer")
    with col2:
        if st.button("🔄 Refresh Now", type="primary", key="refresh_customer"):
            st.rerun()
    
    if auto_refresh:
        time.sleep(2)
        st.rerun()

# ─── ROUTER ──────────────────────────────────────────────────────
if st.session_state.page == "landing":
    show_landing()
elif st.session_state.page == "login_bank":
    show_login("bank")
elif st.session_state.page == "login_customer":
    show_login("customer")
elif st.session_state.page == "dashboard":
    if st.session_state.role == "bank":
        show_bank_dashboard()
    else:
        show_customer_dashboard()
        