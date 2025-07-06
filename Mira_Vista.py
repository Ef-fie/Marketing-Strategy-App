import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Mira Vista Segmentation",
    page_icon="🛍️",
    layout="centered"
)

# App title and intro
st.title("🛍️ Mira Vista Mall - Customer Segmentation Dashboard")
st.markdown(
    "Welcome to the Mira Vista marketing insights tool. "
    "Explore customer clusters and tailor your marketing strategy for maximum impact."
)

# Load the clustered Excel dataset
df = pd.read_excel("C:/Users/effie/OneDrive - Strathmore University/Documents/Mall_Customers.xlsx")

# Showing a colorful scatter plot of the clusters
st.subheader("Customer Clusters")

fig, ax = plt.subplots()
scatter = ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
    cmap='Set2',
    s=80,
    edgecolors='black'
)
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1–100)")
ax.set_title("Customer Segments at Mira Vista")
st.pyplot(fig)

# Showing average traits for each cluster group
st.subheader("Cluster Insights")
summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(1)
st.dataframe(summary)

# Interactive Profile Selector
st.markdown("---")
st.subheader("🧠 Explore Cluster Profiles & Strategy")

cluster_profiles = {
    0: {
        "name": "Practical Professionals",
        "description": (
            "Older customers with an average age of **52**, earning about **$47k**, and showing moderate spending behavior (~**39**). "
            "They are dependable, intentional shoppers who value trust and practicality over trends."
        ),
        "strategies": [
            "🛍️ Introduce practical product bundles and value packs.",
            "🎁 Launch a loyalty program that rewards consistent shoppers.",
            "📧 Send regular emails with curated value offerings and lifestyle tips.",
            "🕰️ Offer relaxed shopping hours and personalized in-store assistance."
        ],
        "bgcolor": "#E8F5E9"
    },
    1: {
        "name": "Affluent Minimalists",
        "description": (
            "Middle-aged shoppers (avg. age **40**) with the **highest income (~$91k)** and the **lowest spending score (~20)**. "
            "These selective customers may prefer other upscale experiences and invest in fewer, more intentional purchases."
        ),
        "strategies": [
            "💎 Offer exclusive invite-only shopping experiences.",
            "🛍️ Highlight premium collections with limited availability.",
            "📬 Use elegant, minimalist email campaigns tailored to their taste.",
            "🌐 Promote personal styling services and concierge options."
        ],
        "bgcolor": "#F3E5F5"
    },
    2: {
        "name": "Vibrant Spenders",
        "description": (
            "The youngest group (avg. age **28**), with a solid income (~**$60k**) and a high spending score (~**69**). "
            "They are energetic, impulsive, trend-following shoppers who love novelty and social buzz."
        ),
        "strategies": [
            "🔥 Launch flash sales and seasonal campaigns with bold visuals.",
            "📲 Partner with influencers and promote content on social media.",
            "🎯 Gamify shopping through reward challenges and QR hunts across the mall.",
            "🧢 Feature trending brands and pop-up experiences designed for discovery."
        ],
        "bgcolor": "#FFF3E0"
    }
}

# Selection and button UI
selected_cluster = st.selectbox("Choose a Cluster (0, 1 or 2):", options=sorted(cluster_profiles.keys()))
show = st.button("Submit")

if show:
    data = cluster_profiles[selected_cluster]
    st.markdown(
        f"""
        <div style='background-color:{data["bgcolor"]}; padding:20px; border-radius:10px; margin-top:15px'>
            <h3 style='color:#333;'>Cluster {selected_cluster}: {data["name"]}</h3>
            <p>{data["description"]}</p>
            <h4>Recommended Marketing Tactics:</h4>
            <ul>
                {''.join(f"<li>{s}</li>" for s in data["strategies"])}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    #  Mira Vista Custom Styling
st.markdown(
    """
    <style>
    /* Set a clean white background */
    .stApp {
        background-color: #FFFFFF;
        color: #333333;  /* Cloud Grey text */
        font-family: 'Segoe UI', sans-serif;
    }

    /* Headers (Midnight Blue) */
    h1, h2, h3, h4 {
        color: #003366;
    }

    /* Primary Buttons */
    .stButton > button {
        background-color: #B76E79;  /* Rose Gold */
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 0 6px rgba(183, 110, 121, 0.4);
    }

    .stButton > button:hover {
        background-color: #A05260;
        color: #ffffff;
    }

    /* Tables and DataFrames */
    .css-1d391kg, .css-1n76uvr {
        background-color: #F0F2F5;  /* Cloud Grey panels */
        color: #333333;
        border-radius: 8px;
        box-shadow: 0px 0px 6px rgba(0,0,0,0.1);
    }

    /* Tabs */
    div[data-baseweb="tab"] {
        background-color: #E6EBF0;
        color: #003366;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
    }

    /* Selectboxes and Text Inputs */
    .stSelectbox, .stTextInput > div, .stTextArea > div {
        background-color: #F8F9FB;
        color: #333333;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background-color: #B76E79;  /* Rose Gold */
        border-radius: 10px;
    }

    /* Hide default footer */
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Floating Navigation Bar
st.markdown(
    """
    <style>
    /* Sticky top bar with midnight blue */
    .nav-bar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #003366;  /* Midnight Blue */
        color: white;
        padding: 12px 40px;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }

    .nav-title {
        font-size: 20px;
        font-weight: 600;
    }

    .stApp {
        padding-top: 70px !important;  /* Push content below nav bar */
    }
    </style>

    <div class="nav-bar">
        <div class="nav-title">Mira Vista Dashboard</div>
        <div style="font-size:14px;">✨ Shop Smarter · Market Brighter</div>
    </div>
    """,
    unsafe_allow_html=True
)