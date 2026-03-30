import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ---------- BACKGROUND STYLE ----------
import base64

def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Dark overlay for readability */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.6);
            z-index: 0;
        }}

        .main {{
            position: relative;
            z-index: 1;
        }}

        h1, h2, h3, h4, h5, h6, p, label {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function
set_bg("bg.png")
# ---------- HERO SECTION ----------
st.markdown(
    """
    <h1 style='font-size:60px;'>⚡ Energy Efficiency Prediction App</h1>
    <p style='font-size:20px;'>Upload your dataset, compare models, and predict results instantly.</p>
    <br><br>
    """,
    unsafe_allow_html=True
)

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file is not None:

    # Read file
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.success("File uploaded successfully ✅")

    # Preview
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    # Select target and features
    target = st.selectbox("Select Target Column", data.columns)
    features = st.multiselect("Select Feature Columns", data.columns)

    if target and features:

        X = data[features]
        y = data[target]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Models
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor()
        }

        results = {}

        # Train all models
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            results[name] = score

        # Show results
        st.subheader("📊 Model Comparison")
        for name, score in results.items():
            st.write(f"{name}: {score:.4f}")

        # Visual graph
        results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        results_df = results_df.set_index("Model")

        st.subheader("📈 Visual Comparison")
        st.bar_chart(results_df)

        # Best model
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]

        st.success(f"🏆 Best Model: {best_model_name}")
        st.write(f"Best Accuracy: {results[best_model_name]:.4f}")

        # Prediction
        st.subheader("🔮 Make Prediction")

        user_input = []
        for col in features:
            val = st.number_input(f"Enter value for {col}", value=0.0)
            user_input.append(val)

        if st.button("Predict"):
            prediction = best_model.predict([user_input])
            st.success(f"Predicted Value: {prediction[0]}")

else:
    st.info("Please upload a CSV or Excel file to continue")