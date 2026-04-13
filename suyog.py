"""
Exam Intelligence App — TCET Edition
Thakur College of Engineering & Technology, Mumbai
Integrated Single File: Backend + Streamlit Frontend
"""

# ============================================================
# IMPORTS
# ============================================================
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from openai import OpenAI

# ============================================================
# DATABASE
# ============================================================

question_bank = {
    ("1st", "Semester 1"): {
        "Maths": [
            "Solve: \u222bx dx",
            "Explain Pythagoras theorem"
        ],
        "Physics": [
            "State Newton's 2nd law",
            "Explain Bernoulli's principle"
        ]
    },
    ("1st", "Semester 2"): {
        "Computer Science": [
            "Explain OOP concepts",
            "What is time complexity of binary search?"
        ],
        "Maths": [
            "Find eigenvalues of a 2x2 matrix"
        ]
    },
    ("2nd", "Semester 3"): {
        "Physics": [
            "Derive equation for kinetic energy",
            "Explain Bernoulli's principle with example"
        ]
    }
}

users = {
    "student1":  {"password": "123", "role": "Student"},
    "teacher1":  {"password": "123", "role": "Teacher"},
    "examcell1": {"password": "123", "role": "Exam Cell"}
}

# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")   # \U0001f534 Replace with your key

# ============================================================
# ML MODEL
# ============================================================

_train_questions = [
    "What is 2+2?",
    "Explain Newton's laws",
    "Derive Schr\u00f6dinger equation",
    "Define variable in Python",
    "Explain OOP concepts",
    "Analyze time complexity of quicksort"
]
_labels = [0, 1, 2, 0, 1, 2]

_vectorizer = TfidfVectorizer()
_X = _vectorizer.fit_transform(_train_questions)
_model = LogisticRegression()
_model.fit(_X, _labels)

# ============================================================
# BACKEND FUNCTIONS
# ============================================================

def get_question_bank(year, semester):
    return question_bank.get((year, semester), {})

def detect_repetition(new_question, old_questions):
    if not old_questions:
        return 0.0
    vec = TfidfVectorizer()
    all_q = old_questions + [new_question]
    tfidf = vec.fit_transform(all_q)
    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
    return float(similarity.max())

def predict_difficulty(question):
    vec = _vectorizer.transform([question])
    pred = _model.predict(vec)[0]
    return ["Easy", "Medium", "Hard"][pred]

def generate_ai_questions(subject, difficulty, old_questions, num_questions=2):
    prompt = (
        f"Generate {num_questions} exam questions for subject: {subject}\n"
        f"Difficulty: {difficulty}\n\n"
        "Rules:\n- Exam oriented\n- No repetition\n"
        "- Output ONLY bullet points (one question per line)"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content
        questions = [q.strip("- ").strip() for q in text.split("\n") if q.strip()]
        final = [q for q in questions if detect_repetition(q, old_questions) < 0.7]
        return final[:num_questions]
    except Exception as e:
        st.warning(f"AI generation skipped: {e}")
        return []

def generate_question_paper(year, semester, difficulty):
    subjects = get_question_bank(year, semester)
    if not subjects:
        return None
    paper = f"QUESTION PAPER \u2014 {year} Year | {semester} | Difficulty: {difficulty}\n"
    paper += "=" * 60 + "\n"
    for subject, questions in subjects.items():
        paper += f"\n--- {subject} ---\n"
        selected = random.sample(questions, min(2, len(questions)))
        for q in selected:
            level = predict_difficulty(q)
            if difficulty == level or difficulty == "Medium":
                paper += f"  \u2022 {q}  [{level}]\n"
        ai_qs = generate_ai_questions(subject, difficulty, questions, num_questions=2)
        for q in ai_qs:
            paper += f"  \u2022 {q}  [AI \u2013 {difficulty}]\n"
    return paper

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TCET Exam Intelligence",
    page_icon="\U0001f393",
    layout="wide"
)

# ============================================================
# SESSION STATE
# ============================================================

for key in ["role", "year", "semester", "logged_in"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# LOGIN PAGE \u2014 TCET ENHANCED
# ============================================================

if not st.session_state.get("logged_in"):

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,600&family=Inter:wght@300;400;500;600&display=swap');

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }

    .stApp {
        background: #080e1c;
        font-family: 'Inter', sans-serif;
    }

    /* \u2500\u2500 TOP ACCENT BAR \u2500\u2500 */
    .accent-bar {
        height: 4px;
        background: linear-gradient(90deg, #c8960c 0%, #f5c842 40%, #c8960c 70%, #8a6500 100%);
    }

    /* \u2500\u2500 SPLIT LAYOUT \u2500\u2500 */
    .login-wrap {
        display: grid;
        grid-template-columns: 1fr 1fr;
        min-height: 100vh;
    }

    /* LEFT PANEL */
    .left-panel {
        background: linear-gradient(160deg, #0b1530 0%, #071028 60%, #060d1f 100%);
        padding: 3.5rem 3rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        border-right: 1px solid rgba(200,150,12,0.15);
        position: relative;
        overflow: hidden;
    }
    .left-panel::before {
        content: '';
        position: absolute;
        top: -120px; right: -120px;
        width: 380px; height: 380px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(200,150,12,0.07) 0%, transparent 70%);
        pointer-events: none;
    }
    .left-panel::after {
        content: '';
        position: absolute;
        bottom: -80px; left: -80px;
        width: 260px; height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(99,179,237,0.05) 0%, transparent 70%);
        pointer-events: none;
    }

    /* TCET LOGO AREA */
    .tcet-header {
        display: flex;
        align-items: center;
        gap: 1.25rem;
        margin-bottom: 2.5rem;
    }
    .tcet-name h2 {
        font-family: 'Playfair Display', serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #f5c842;
        margin: 0 0 3px;
        letter-spacing: 0.5px;
        line-height: 1.2;
    }
    .tcet-name p {
        font-size: 11px;
        color: #7a9abf;
        margin: 0;
        font-weight: 400;
        letter-spacing: 0.3px;
    }

    /* HERO TEXT */
    .hero-text h1 {
        font-family: 'Playfair Display', serif;
        font-size: clamp(1.9rem, 3vw, 2.6rem);
        color: #eef4ff;
        font-weight: 700;
        line-height: 1.2;
        margin: 0 0 0.5rem;
    }
    .hero-text h1 em {
        font-style: italic;
        color: #f5c842;
    }
    .hero-text .tagline {
        font-size: 13.5px;
        color: #6888aa;
        font-weight: 300;
        margin: 0 0 2rem;
        line-height: 1.6;
    }

    /* ACCREDITATION BADGES */
    .badge-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 2rem;
    }
    .acc-badge {
        background: rgba(200,150,12,0.08);
        border: 1px solid rgba(200,150,12,0.25);
        border-radius: 6px;
        padding: 5px 12px;
        font-size: 11px;
        font-weight: 600;
        color: #d4a82a;
        letter-spacing: 0.6px;
    }

    /* FEATURE CARDS */
    .feature-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 2.5rem;
    }
    .feat-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 14px;
        transition: border-color 0.2s;
    }
    .feat-card:hover { border-color: rgba(200,150,12,0.3); }
    .feat-icon {
        font-size: 18px;
        margin-bottom: 6px;
        display: block;
    }
    .feat-title {
        font-size: 12px;
        font-weight: 600;
        color: #c8d8ee;
        margin-bottom: 2px;
    }
    .feat-sub {
        font-size: 11px;
        color: #4a6888;
        line-height: 1.4;
    }

    /* FOOTER */
    .left-footer {
        font-size: 11px;
        color: #2d4460;
        border-top: 1px solid rgba(255,255,255,0.05);
        padding-top: 1rem;
    }

    /* RIGHT PANEL */
    .right-panel {
        background: #0d1525;
        padding: 3.5rem 3rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .form-heading {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: #dce8f8;
        font-weight: 600;
        margin: 0 0 0.3rem;
    }
    .form-sub {
        font-size: 13px;
        color: #4a6888;
        margin: 0 0 2rem;
    }

    /* ROLE SELECTOR */
    .role-selector {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-bottom: 1.5rem;
    }
    .role-tile {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 12px 8px;
        text-align: center;
        cursor: pointer;
        transition: all 0.18s;
    }
    .role-tile:hover {
        border-color: rgba(200,150,12,0.4);
        background: rgba(200,150,12,0.05);
    }
    .role-tile.active {
        border-color: #c8960c;
        background: rgba(200,150,12,0.1);
    }
    .role-tile .rt-icon { font-size: 20px; display: block; margin-bottom: 5px; }
    .role-tile .rt-label { font-size: 11.5px; color: #7a9abf; font-weight: 500; }
    .role-tile.active .rt-label { color: #f5c842; }

    /* SECTION LABEL */
    .slabel {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: #3d6080;
        margin-bottom: 0.7rem;
        display: block;
    }

    /* STREAMLIT OVERRIDES */
    .stTextInput > label, .stSelectbox > label {
        color: #5a7a9a !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput input {
        background: #070e1c !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 8px !important;
        color: #dce8f8 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
    }
    .stTextInput input:focus {
        border-color: #c8960c !important;
        box-shadow: 0 0 0 3px rgba(200,150,12,0.12) !important;
    }
    .stTextInput input::placeholder { color: #2d4460 !important; }
    .stSelectbox > div > div {
        background: #070e1c !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 8px !important;
        color: #dce8f8 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* SIGN IN BUTTON */
    .stButton > button {
        background: linear-gradient(135deg, #9a6e00 0%, #c8960c 50%, #e8b020 100%) !important;
        color: #0d1525 !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        letter-spacing: 0.8px !important;
        text-transform: uppercase !important;
        padding: 0.7rem 2rem !important;
        width: 100% !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 20px rgba(200,150,12,0.3) !important;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 30px rgba(200,150,12,0.5) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* DEMO CARD */
    .demo-card {
        background: rgba(200,150,12,0.05);
        border: 1px solid rgba(200,150,12,0.12);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 12px;
        color: #5a7a9a;
        line-height: 2;
        margin-top: 1rem;
    }
    .demo-card b { color: #9ab4cc; }

    /* ROLE HINT */
    .role-hint-box {
        margin-top: 0.75rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 9px 14px;
        font-size: 13px;
        color: #7a9abf;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* DIVIDER */
    .form-divider {
        height: 1px;
        background: rgba(255,255,255,0.05);
        margin: 1.5rem 0;
    }

    /* UNIVERSITY TAG */
    .univ-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 11px;
        color: #3d5a78;
        margin-top: 1.5rem;
    }
    .univ-dot { width: 5px; height: 5px; border-radius: 50%; background: #3d5a78; }

    .stAlert { border-radius: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

    # \u2500\u2500 Gold top bar \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    # \u2500\u2500 Two-column layout \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    col_left, col_right = st.columns([1, 1], gap="small")

    # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
    # LEFT PANEL
    # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
    with col_left:

        # TCET Real Logo + Name
        st.markdown(f"""
        <div class="tcet-header">
          <img src="data:image/jpeg;base64,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" 
               alt="TCET Logo" 
               style="width:90px;height:105px;object-fit:contain;filter:drop-shadow(0 4px 12px rgba(200,150,12,0.35));" />
          <div class="tcet-name">
            <h2>Thakur College of Engineering<br>&amp; Technology</h2>
            <p>Kandivali (E), Mumbai &#8212; Est. 2001</p>
          </div>
        </div>

        <div class="hero-text">
          <h1>Exam <em>Intelligence</em><br>Platform</h1>
          <p class="tagline">AI-powered question generation, smart difficulty prediction,<br>and role-based academic dashboards &#8212; built for TCET.</p>
        </div>

        <div class="badge-row">
          <span class="acc-badge">NAAC A Grade</span>
          <span class="acc-badge">NBA Accredited</span>
          <span class="acc-badge">ISO 9001:2015</span>
          <span class="acc-badge">AICTE Approved</span>
          <span class="acc-badge">Univ. of Mumbai</span>
        </div>

        <div class="feature-grid">
          <div class="feat-card">
            <span class="feat-icon">&#129302;</span>
            <div class="feat-title">AI Question Generator</div>
            <div class="feat-sub">GPT-4o powered unique exam questions</div>
          </div>
          <div class="feat-card">
            <span class="feat-icon">&#128202;</span>
            <div class="feat-title">Difficulty Prediction</div>
            <div class="feat-sub">ML model classifies Easy / Medium / Hard</div>
          </div>
          <div class="feat-card">
            <span class="feat-icon">&#128105;&#8205;&#127979;</span>
            <div class="feat-title">Teacher Dashboard</div>
            <div class="feat-sub">Assign, push &amp; manage papers</div>
          </div>
          <div class="feat-card">
            <span class="feat-icon">&#127970;</span>
            <div class="feat-title">Exam Cell Portal</div>
            <div class="feat-sub">Centralized QB &amp; analytics</div>
          </div>
        </div>

        <div class="left-footer">
          Affiliated to University of Mumbai &nbsp;&middot;&nbsp; NIRF Ranked &nbsp;&middot;&nbsp; Kandivali, Mumbai
        </div>
        """, unsafe_allow_html=True)

    # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
    # RIGHT PANEL \u2014 FORM
    # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
    with col_right:

        st.markdown("""
        <div style="max-width:420px; margin: 0 auto; padding: 2rem 0;">
          <div class="form-heading">Welcome back</div>
          <div class="form-sub">Sign in to access your TCET dashboard</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span class="slabel">Your Role</span>', unsafe_allow_html=True)

        # Role visual hint
        username_val = st.session_state.get("_preview_user", "")
        role_icons   = {"Student": "\U0001f393", "Teacher": "\U0001f469\u200d\U0001f3eb", "Exam Cell": "\U0001f3e2"}
        role_colors  = {"Student": "#2ecc71", "Teacher": "#f39c12", "Exam Cell": "#c084fc"}

        st.markdown("""
        <div class="role-selector" id="roleSel">
          <div class="role-tile" onclick="">
            <span class="rt-icon">\U0001f393</span>
            <span class="rt-label">Student</span>
          </div>
          <div class="role-tile" onclick="">
            <span class="rt-icon">\U0001f469\u200d\U0001f3eb</span>
            <span class="rt-label">Teacher</span>
          </div>
          <div class="role-tile" onclick="">
            <span class="rt-icon">\U0001f3e2</span>
            <span class="rt-label">Exam Cell</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="form-divider"></div>', unsafe_allow_html=True)

        st.markdown('<span class="slabel">Credentials</span>', unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter your username", label_visibility="collapsed")
        st.markdown('<span class="slabel" style="margin-top:0.5rem; display:block;">Password</span>', unsafe_allow_html=True)
        password = st.text_input("Password", type="password", placeholder="Enter your password", label_visibility="collapsed")

        # Live role hint
        if username and username in users:
            detected = users[username]["role"]
            icon     = role_icons.get(detected, "\U0001f464")
            color    = role_colors.get(detected, "#63b3ed")
            st.markdown(f"""
            <div class="role-hint-box">
              <span style="font-size:18px">{icon}</span>
              <span>Detected role: <b style="color:{color}">{detected}</b></span>
            </div>
            """, unsafe_allow_html=True)
        elif username:
            st.markdown("""
            <div class="role-hint-box">
              <span style="font-size:18px">\u26a0\ufe0f</span>
              <span style="color:#e05252">Username not found</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="form-divider"></div>', unsafe_allow_html=True)
        st.markdown('<span class="slabel">Academic Context</span>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            year_choice = st.selectbox("Year", ["1st", "2nd", "3rd", "4th"], label_visibility="collapsed")
        with c2:
            semester_choice = st.selectbox(
                "Semester",
                ["Semester 1", "Semester 2", "Semester 3", "Semester 4",
                 "Semester 5", "Semester 6", "Semester 7", "Semester 8"],
                label_visibility="collapsed"
            )

        st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)

        if st.button("Sign In to TCET Portal \u2192"):
            if username in users and users[username]["password"] == password:
                st.session_state["role"]      = users[username]["role"]
                st.session_state["logged_in"] = True
                st.session_state["year"]      = year_choice
                st.session_state["semester"]  = semester_choice
                st.success(f"\u2705 Welcome! Redirecting to your {st.session_state['role']} dashboard\u2026")
                st.rerun()
            else:
                st.error("\u26a0\ufe0f Invalid credentials. Please check your username and password.")

        st.markdown("""
        <div class="demo-card">
          <div style="font-size:10px; font-weight:600; letter-spacing:1.5px; color:#3d5a78; margin-bottom:6px;">DEMO CREDENTIALS</div>
          \U0001f393 &nbsp;<b>student1</b> / 123 &nbsp;&middot;&nbsp; Student Portal<br>
          \U0001f469\u200d\U0001f3eb &nbsp;<b>teacher1</b> / 123 &nbsp;&middot;&nbsp; Teacher Dashboard<br>
          \U0001f3e2 &nbsp;<b>examcell1</b> / 123 &nbsp;&middot;&nbsp; Exam Cell Portal
        </div>

        <div class="univ-tag">
          <div class="univ-dot"></div>
          Affiliated to University of Mumbai &nbsp;|&nbsp; Academic Year 2024\u201325
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ============================================================
# SIDEBAR (post-login)
# ============================================================

with st.sidebar:
    st.markdown(f"""
    <div style="padding: 0.5rem 0 1rem;">
      <div style="font-size:13px; color:#c8960c; font-weight:600; letter-spacing:0.5px;">TCET PORTAL</div>
      <div style="font-size:16px; font-weight:600; color:#dce8f8; margin-top:4px;">
        {st.session_state.get('role', '')}
      </div>
      <div style="font-size:12px; color:#4a6888; margin-top:2px;">
        {st.session_state.get('year', '')} Year &nbsp;&middot;&nbsp; {st.session_state.get('semester', '')}
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    if st.button("\U0001f6aa Logout"):
        for key in ["role", "year", "semester", "logged_in"]:
            st.session_state[key] = None
        st.rerun()

role     = st.session_state.get("role")
year     = st.session_state.get("year")
semester = st.session_state.get("semester")

# ============================================================
# STUDENT DASHBOARD
# ============================================================

if role == "Student":
    st.header("\U0001f393 Student Dashboard")
    st.subheader("\U0001f4dd Practice Test")
    difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"], key="student_difficulty")
    if st.button("Generate Practice Test", key="student_generate"):
        with st.spinner("Generating practice test\u2026"):
            paper = generate_question_paper(year, semester, difficulty)
        if paper:
            st.success("Practice test generated successfully!")
            st.text(paper)
            st.download_button("\U0001f4e5 Download (TXT)", data=paper,
                file_name=f"Student_{year}_{semester}_{difficulty}_paper.txt",
                mime="text/plain", key="student_paper_txt")
            st.download_button("\U0001f4e5 Download (Word)", data=paper,
                file_name=f"Student_{year}_{semester}_{difficulty}_paper.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="student_paper_docx")
        else:
            st.warning("No questions found for this Year & Semester combination.")

    st.divider()
    st.subheader("\U0001f4d6 Formula Sheet")
    with st.expander("View Formula Sheet"):
        st.markdown("**Algebra**")
        st.latex(r"(a+b)^2 = a^2 + 2ab + b^2")
        st.latex(r"a^2 - b^2 = (a-b)(a+b)")
        st.markdown("**Calculus**")
        st.latex(r"\frac{d}{dx}(x^n) = nx^{n-1}")
        st.latex(r"\int x^n\,dx = \frac{x^{n+1}}{n+1} + C")
        st.markdown("**Physics**")
        st.latex(r"F = ma")
        st.latex(r"E = mc^2")
        st.latex(r"KE = \tfrac{1}{2}mv^2")

    st.divider()
    st.subheader("\U0001f0cf Flashcards")
    with st.expander("View Flashcards"):
        flashcards = {
            "Expand (a+b)\u00b2": "a\u00b2 + 2ab + b\u00b2",
            "Area of a circle?": "\u03c0r\u00b2",
            "sin\u00b2\u03b8 + cos\u00b2\u03b8 = ?": "1",
            "Derivative of x\u207f?": "n\u00b7x\u207f\u207b\u00b9",
            "Newton's Second Law?": "F = ma",
        }
        for question, answer in flashcards.items():
            with st.expander(f"Q: {question}"):
                st.write(f"A: {answer}")

    st.divider()
    st.subheader("\U0001f4ca Progress Tracking")
    st.line_chart([10, 20, 15, 30])

# ============================================================
# TEACHER DASHBOARD
# ============================================================

elif role == "Teacher":
    st.header("\U0001f469\u200d\U0001f3eb Teacher Dashboard")
    st.subheader("\U0001f4cb Create Assignment")
    uploaded_assignment = st.file_uploader("Upload Assignment", key="teacher_upload")
    if st.button("Assign to Students", key="teacher_assign"):
        st.success("Assignment created and assigned!") if uploaded_assignment else st.warning("Please upload a file first.")

    st.divider()
    st.subheader("\U0001f4c8 Student Analytics")
    st.line_chart([5, 15, 25, 35])

    st.divider()
    st.subheader("\U0001f4dd Question Paper Generator")
    difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"], key="teacher_difficulty")
    if st.button("Generate Paper", key="teacher_generate"):
        with st.spinner("Generating paper\u2026"):
            paper = generate_question_paper(year, semester, difficulty)
        if paper:
            st.success("Paper generated!")
            st.text(paper)
            st.download_button("\U0001f4e5 Download (TXT)", data=paper,
                file_name=f"Teacher_{year}_{semester}_{difficulty}_paper.txt",
                mime="text/plain", key="teacher_paper_txt")
        else:
            st.warning("No questions found for this Year & Semester combination.")

    st.divider()
    st.subheader("\U0001f52c Case Study Questions")
    if st.button("Generate Case Study Questions", key="teacher_case_study"):
        st.info("Case study questions generated using AI (demo).")

    st.divider()
    st.subheader("\U0001f4e4 Push Assignment to Classes")
    uploaded_push = st.file_uploader("Upload Assignment for Distribution", key="teacher_push_upload")
    if st.button("Push to Classes", key="teacher_push"):
        st.success("Assignment pushed to all classes!") if uploaded_push else st.warning("Please upload a file first.")

# ============================================================
# EXAM CELL DASHBOARD
# ============================================================

elif role == "Exam Cell":
    st.header("\U0001f3e2 Exam Cell Dashboard")
    st.subheader("\U0001f4c2 Centralized Question Bank")
    if (year, semester) in question_bank:
        subjects       = question_bank[(year, semester)]
        subject_choice = st.selectbox("Select Subject", list(subjects.keys()), key="ec_qb_subject")
        if st.button("View Question Bank", key="ec_qb_view"):
            st.success(f"Questions for {subject_choice} \u2014 {year}, {semester}:")
            for i, q in enumerate(subjects[subject_choice], 1):
                st.write(f"{i}. {q}")
            st.download_button("\U0001f4e5 Download QB (TXT)",
                data="\n".join(subjects[subject_choice]),
                file_name=f"{year}_{semester}_{subject_choice}_QB.txt",
                mime="text/plain", key="ec_qb_download")
    else:
        st.warning("No Question Bank available for this Year & Semester yet.")

    st.divider()
    st.subheader("\U0001f4dd Question Paper Generator")
    difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"], key="ec_difficulty")
    if st.button("Generate Paper", key="ec_generate"):
        with st.spinner("Generating paper\u2026"):
            paper = generate_question_paper(year, semester, difficulty)
        if paper:
            st.success("Paper generated!")
            st.text(paper)
            st.download_button("\U0001f4e5 Download (TXT)", data=paper,
                file_name=f"ExamCell_{year}_{semester}_{difficulty}_paper.txt",
                mime="text/plain", key="ec_paper_txt")
        else:
            st.warning("No questions found for this Year & Semester combination.")

    st.divider()
    st.subheader("\U0001f4ca Analytics Dashboard")
    st.line_chart([12, 18, 22, 28])

    st.divider()
    st.subheader("\U0001f4c4 Exam Model Paper")
    uploaded_pattern = st.file_uploader("Upload paper pattern", key="ec_model_upload")
    if st.button("Publish to Classes", key="ec_publish"):
        st.success("Exam model paper published!") if uploaded_pattern else st.warning("Please upload a file first.")