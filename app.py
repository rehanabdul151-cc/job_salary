# =========================
# IMPORT LIBRARIES
# =========================
import streamlit as st
import pickle
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    background-color: #f5f7ff;
}

.login-box {
    padding: 30px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
}

.title {
    text-align: center;
    color: #4a4aff;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "users" not in st.session_state:
    st.session_state.users = {}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = ""

# =========================
# LOAD FILES
# =========================
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =========================
# HELPER FUNCTION
# =========================
def get_options(prefix):
    opts = [col.replace(prefix, "") for col in columns if col.startswith(prefix)]
    return sorted(list(set(opts)))

# Options
job_options = ["Other"] + get_options("job_title_")
edu_options = ["Other"] + get_options("education_level_")
loc_options = ["Other"] + get_options("location_")
ind_options = ["Other"] + get_options("industry_")
company_options = ["Other"] + get_options("company_size_")
remote_options = ["Other"] + get_options("remote_work_")

# =========================
# LOGIN / SIGNUP PAGE
# =========================
if not st.session_state.logged_in:

    st.markdown(
        "<div class='title'>💼 Salary Predictor</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>AI Powered Salary Prediction System</div>",
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["🔑 Login", "📝 Signup"])

    # =========================
    # LOGIN TAB
    # =========================
    with tab1:

        st.markdown("### Welcome Back 👋")

        login_user = st.text_input(
            "Username",
            key="login_user"
        )

        login_pass = st.text_input(
            "Password",
            type="password",
            key="login_pass"
        )

        if st.button("Login"):

            if (
                login_user in st.session_state.users and
                st.session_state.users[login_user] == login_pass
            ):

                st.session_state.logged_in = True
                st.session_state.current_user = login_user

                st.success("✅ Login Successful")

                # Redirect to app page
                st.rerun()

            else:
                st.error("❌ Invalid Username or Password")

    # =========================
    # SIGNUP TAB
    # =========================
    with tab2:

        st.markdown("### Create New Account 🚀")

        signup_user = st.text_input(
            "Create Username",
            key="signup_user"
        )

        signup_pass = st.text_input(
            "Create Password",
            type="password",
            key="signup_pass"
        )

        confirm_pass = st.text_input(
            "Confirm Password",
            type="password",
            key="confirm_pass"
        )

        if st.button("Create Account"):

            if signup_user == "" or signup_pass == "":
                st.warning("⚠ Please fill all fields")

            elif signup_user in st.session_state.users:
                st.error("⚠ Username already exists")

            elif signup_pass != confirm_pass:
                st.error("⚠ Passwords do not match")

            else:
                # Save user
                st.session_state.users[signup_user] = signup_pass

                # Auto Login After Signup
                st.session_state.logged_in = True
                st.session_state.current_user = signup_user

                st.success("✅ Account Created Successfully")

                # Redirect to salary page
                st.rerun()

# =========================
# MAIN APP
# =========================
else:

    # =========================
    # HEADER
    # =========================
    col1, col2 = st.columns([4, 1])

    with col1:
        st.title("💼 Salary Prediction App")

    with col2:
        st.write("")
        st.write(f"👤 {st.session_state.current_user}")

    st.markdown("---")

    # =========================
    # USER INPUTS
    # =========================
    st.subheader("📋 Enter Your Details")

    exp = st.number_input(
        "Experience (Years)",
        0,
        30
    )

    skills = st.number_input(
        "Skills Count",
        0,
        50
    )

    cert = st.number_input(
        "Certifications",
        0,
        20
    )

    job = st.selectbox("Job Role", job_options)
    edu = st.selectbox("Education", edu_options)
    loc = st.selectbox("Location", loc_options)
    ind = st.selectbox("Industry", ind_options)
    company = st.selectbox("Company Size", company_options)
    remote = st.selectbox("Remote Work", remote_options)

    # =========================
    # CREATE DATAFRAME
    # =========================
    input_dict = {
        "experience_years": exp,
        "skills_count": skills,
        "certifications": cert,
        "job_title": job,
        "education_level": edu,
        "location": loc,
        "industry": ind,
        "company_size": company,
        "remote_work": remote
    }

    input_df = pd.DataFrame([input_dict])

    # =========================
    # FEATURE ENGINEERING
    # =========================
    input_df['exp_squared'] = input_df['experience_years'] ** 2

    input_df['skill_per_exp'] = (
        input_df['skills_count'] /
        (input_df['experience_years'] + 1)
    )

    input_df['cert_per_skill'] = (
        input_df['certifications'] /
        (input_df['skills_count'] + 1)
    )

    input_df['seniority'] = pd.cut(
        input_df['experience_years'],
        bins=[0, 2, 5, 10, 20],
        labels=['Fresher', 'Junior', 'Mid', 'Senior']
    )

    # =========================
    # DUMMIES + ALIGN
    # =========================
    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(
        columns=columns,
        fill_value=0
    )

    # =========================
    # SCALE
    # =========================
    num_cols = [
        'experience_years',
        'skills_count',
        'certifications',
        'exp_squared',
        'skill_per_exp',
        'cert_per_skill'
    ]

    input_df[num_cols] = scaler.transform(
        input_df[num_cols]
    )

    # =========================
    # PREDICT BUTTON
    # =========================
    if st.button("💰 Predict Salary"):

        prediction = model.predict(input_df)

        st.success(
            f"🎉 Predicted Salary: ₹ {int(prediction[0]):,}"
        )

        st.balloons()

    # =========================
    # LOGOUT BUTTON
    # =========================
    st.markdown("---")

    if st.button("🚪 Logout"):

        st.session_state.logged_in = False
        st.session_state.current_user = ""

        st.rerun()