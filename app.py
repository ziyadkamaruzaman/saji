import os
import urllib.parse
import pandas as pd
import streamlit as st
from src.saji.recommender import load_artifacts, recommend

# =========================
# CONFIG
# =========================
ART_DIR = os.path.abspath("artifacts")
ASSETS_DIR = os.path.abspath("assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

APP_TITLE = "Saji"
APP_TAGLINE = "Malaysia's Smart Food Compass"

st.set_page_config(
    page_title=APP_TITLE, 
    page_icon="🍽️", 
    layout="wide"
)

# Clean Light Theme with Warm Accents
st.markdown("""
<style>
    .stApp { background: #f8fafc; color: #1e2937; }
    .hero {
        border-radius: 20px;
        padding: 35px 30px;
        background: white;
        border: 2px solid #f59e0b;
        margin-bottom: 30px;
        text-align: center;
    }
    .hero-image {
        width: 100%;
        max-height: 240px;
        object-fit: cover;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .mm-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .mm-name { font-size: 1.25rem; font-weight: 600; color: #1e2937; }
    .mm-meta { color: #64748b; font-size: 0.97rem; margin: 8px 0; }
    .badge {
        display: inline-block;
        padding: 6px 13px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-good { background: #14532d; color: #86efac; }
    .badge-bad { background: #7f1d1d; color: #fda4af; }
    .stButton > button {
        background-color: #f59e0b !important;
        color: white !important;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #d97706 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "go" not in st.session_state:
    st.session_state.go = False
if "qs_query" not in st.session_state:
    st.session_state.qs_query = None

# =========================
# LOAD DATA
# =========================
@st.cache_resource
def _load():
    return load_artifacts(ART_DIR)

dedup, tfidf, X = _load()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Search")
    query_input = st.text_input(
        "What are you craving?", 
        placeholder="nasi kandar, ayam penyet...",
        key="query_input"
    )
    city = st.text_input("City / Area", placeholder="Kuala Lumpur, Penang...")

    if st.button("Recommend", use_container_width=True, type="primary"):
        st.session_state.go = True
        st.session_state.qs_query = None

    with st.expander("Filters", expanded=False):
        topk = st.slider("Results", 5, 25, 10)
        food_type = st.selectbox(
            "Cuisine", 
            ["Any", "Malay", "Mamak", "Chinese", "Indian", "Japanese", "Western"], 
            index=0
        )
        food_type = "" if food_type == "Any" else food_type
        
        st.markdown("**Halal**")
        halal_only = st.toggle("Halal-friendly only", value=False)
        exclude_pork = st.toggle("Exclude pork & alcohol", value=False)
        min_rating = st.select_slider(
            "Minimum rating", 
            options=[0.0, 4.0, 4.3, 4.5], 
            value=0.0
        )

# =========================
# HERO
# =========================
st.markdown(f"""
<div class="hero">
    <img src="https://picsum.photos/id/292/900/240" class="hero-image" alt="Malaysian Food">
    <h1 class="hero-title">{APP_TITLE}</h1>
    <p class="hero-sub">{APP_TAGLINE}</p>
    <p style="color:#475569; margin-top:12px;">Discover better places to eat — powered by real Malaysian reviews</p>
</div>
""", unsafe_allow_html=True)

# =========================
# POPULAR SEARCHES (Using your own images)
# =========================
st.markdown("**Popular Searches**")
cols = st.columns(3)

# Map popular items to your local asset files
popular_items = [
    ("Nasi Kandar", "nasi kandar", "nasi-kandar.png"),
    ("Ayam Penyet", "ayam penyet", "ayam-penyet.png"),
    ("Mamak", "mamak", "mamak.png"),
    ("Sushi", "sushi", "sushi.png"),
    ("Laksa", "laksa", "laksa.png"),
    ("Burger", "burger", "burger.png")
]

for i, (label, value, filename) in enumerate(popular_items):
    img_path = os.path.join(ASSETS_DIR, filename)
    
    with cols[i % 3]:
        # Use local image if it exists, fallback to placeholder
        if os.path.exists(img_path):
            st.image(img_path, width="stretch")          # New recommended way
        else:
            st.image(f"https://picsum.photos/id/{1000 + i}/300/160", 
                    width="stretch")
        
        if st.button(label, key=f"qs_{i}", use_container_width=True):
            st.session_state.qs_query = value
            st.session_state.go = True
            st.rerun()

# =========================
# RECOMMENDATION LOGIC
# =========================
current_query = st.session_state.qs_query or query_input.strip()

if current_query and st.session_state.go:
    with st.spinner("Finding the best places..."):
        try:
            res = recommend(
                dedup, tfidf, X,
                query=current_query,
                topk=topk,
                city=city.strip() if city.strip() else None,
                food_type=food_type if food_type else None,
                min_rating=min_rating if min_rating > 0 else None,
                halal_only=halal_only,
                exclude_pork_alcohol=exclude_pork
            )
            
            matches = res["matches"]
            st.subheader(f"Results for **{current_query}**")
            
            if len(matches) == 0:
                st.info("No strong matches found. Try adjusting filters or a different search term.")
            else:
                for idx, r in matches.iterrows():
                    name = str(r.get("name", ""))
                    city_out = str(r.get("city", "")).title()
                    rating = f"{r.get('bayes_rating', 0):.1f}"
                    reviews = f"{int(r.get('review_count', 0)):,}"
                    
                    is_halal = bool(r.get("halal_flag", True)) and not bool(r.get("non_halal_flag", False))
                    
                    badges = []
                    if pd.notna(r.get("food_type")) and str(r["food_type"]).strip():
                        badges.append(f'<span class="badge">{r["food_type"]}</span>')
                    badges.append(
                        '<span class="badge badge-good">Halal-friendly</span>' 
                        if is_halal else '<span class="badge badge-bad">Non-halal</span>'
                    )
                    
                    st.markdown(f"""
                    <div class="mm-card">
                        <div class="mm-name">{idx+1}. {name}</div>
                        <div class="mm-meta">{city_out} • {rating} ★ • {reviews} reviews</div>
                        <div>{"".join(badges)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.link_button(
                        "Open in Google Maps",
                        f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote_plus(f'{name} {city_out}')}"
                    )
                    
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
    
    # Reset after showing results
    st.session_state.go = False
    if st.session_state.qs_query:
        st.session_state.qs_query = None

else:
    st.info("Start by typing what you're craving or click on the popular searches above.")

st.caption("Saji ranks based on real reviews and smart scoring. Halal status is indicative — please verify directly.")