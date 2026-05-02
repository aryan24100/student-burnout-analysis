import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Student Burnout Analysis", layout="wide")

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(180deg, #040b14, #02070f);
}
.card {
    background: #0a1a2f;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #0f2d4d;
    box-shadow: 0 0 20px rgba(0, 229, 255, 0.05);
    transition: 0.3s;
}
.card:hover {
    box-shadow: 0 0 25px rgba(0, 229, 255, 0.2);
}
.title {
    font-size: 32px;
    font-weight: 700;
    color: #00e5ff;
}
.stButton>button {
    background: linear-gradient(90deg, #00e5ff, #0077ff);
    color: black;
    font-weight: bold;
    border-radius: 12px;
    height: 50px;
}
.fade {
    animation: fadeIn 0.8s ease-in;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.markdown('<div class="title fade">🧠 Student Burnout Analysis - Student Analytics</div>', unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/student_burnout_dataset.csv")
    return df.drop_duplicates().dropna()

df = load_data()

# -------------------------
# MODEL
# -------------------------
@st.cache_resource
def train_model(df):
    X = df[['sleep_hours','study_hours','screen_time_hours','focus_index','exercise_minutes','social_media_hours']]
    y = df['burnout_level']
    return LinearRegression().fit(X, y)

model = train_model(df)

# -------------------------
# RADAR CHART
# -------------------------
def radar_chart(values):
    labels = ["Sleep", "Study", "Screen", "Focus", "Exercise", "Social"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        line=dict(color='#00e5ff')
    ))
    fig.update_layout(
        polar=dict(bgcolor="#0a1a2f", radialaxis=dict(visible=True, range=[0,100])),
        showlegend=False,
        paper_bgcolor="#0a1a2f",
        font_color="white"
    )
    return fig

# -------------------------
# LAYOUT
# -------------------------
left, right = st.columns([1,1])

# =========================
# LEFT PANEL
# =========================
with left:
    st.markdown('<div class="card fade">', unsafe_allow_html=True)

    st.subheader("Student Profile")

    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    study = st.slider("Study Hours", 0.0, 12.0, 4.5)
    screen = st.slider("Screen Time", 0.0, 12.0, 5.0)
    focus = st.slider("Focus Index", 0.0, 100.0, 50.0)
    exercise = st.slider("Exercise (min)", 0, 150, 60)
    social = st.slider("Social Media", 0.0, 8.0, 3.0)

    predict = st.button("⚡ ANALYSE BURNOUT")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# RIGHT PANEL
# =========================
with right:
    st.markdown('<div class="card fade">', unsafe_allow_html=True)

    st.subheader("Behavioural Radar")

    radar_values = [
        (sleep/12)*100,
        (study/12)*100,
        (screen/12)*100,
        focus,
        (exercise/150)*100,
        (social/8)*100
    ]

    st.plotly_chart(radar_chart(radar_values), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sleep", f"{sleep}h")
    col2.metric("Screen", f"{screen}h")
    col3.metric("Study", f"{study}h")
    col4.metric("Exercise", f"{exercise}m")

    if predict:
        import numpy as np
        pred = model.predict(np.array([[sleep, study, screen, focus, exercise, social]]))
        pred = round(pred, 2)

        st.markdown("---")
        st.success(f"🔥 Final Burnout Score: {pred}")

        # =========================
        # SOLUTIONS
        # =========================

        if pred < 30:
            st.info("🟢 Low Burnout")
            st.markdown("### ✅ Recommendations")
            st.write("- Maintain your current healthy routine")
            st.write("- Keep consistent sleep schedule (7–8 hours)")
            st.write("- Balance study and relaxation time")
            st.write("- Stay physically active")
            st.write("- Practice light mindfulness")
            st.write("- Avoid unnecessary screen usage")

        elif pred < 60:
            st.warning("🟡 Moderate Burnout")
            st.markdown("### ⚠ Recommendations")
            st.write("- Improve sleep schedule")
            st.write("- Reduce screen time before sleep")
            st.write("- Use Pomodoro technique for study")
            st.write("- Take regular breaks")
            st.write("- Increase physical activity")
            st.write("- Manage stress through relaxation")

        else:
            st.error("🔴 High Burnout")
            st.markdown("### 🚨 Immediate Actions Required")
            st.write("- Prioritize sleep immediately")
            st.write("- Reduce academic workload")
            st.write("- Limit screen exposure strictly")
            st.write("- Take frequent mental breaks")
            st.write("- Practice meditation or exercise")
            st.write("- Seek help from mentor/counselor")

    else:
        st.markdown("""
        <div style="text-align:center; padding:40px; color:#5a7fa0;">
        ⚡ Awaiting Input<br><br>
        Adjust sliders and click Analyse
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("Aryan Gupta 1CSE23 | Data Science Project")

try:
    df = load_data()
except:
    st.error("Dataset not found. Please check data folder.")
    