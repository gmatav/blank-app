import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Genetic Analysis Toolkit",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🧬 Genetic Analysis Toolkit")

st.markdown("""
Welcome! Choose a tool:

- 📍 Closest Populations  
- 🧪 2-Way Admixture (optimized)  
- ⚖️ 2-Way Fixed Mixture (test ratios)  
""")

# -----------------------------
# Navigation buttons
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("📍 Closest Populations"):
        st.switch_page("pages/Closest_Populations.py")

    if st.button("🧪 2-Way Admixture"):
        st.switch_page("pages/two_way_admixture.py")

with col2:
    if st.button("⚖️ Fixed Mixture"):
        st.switch_page("pages/Admixture_Fixed.py")

    if st.button("🧪 3-Way Admixture"):
        st.switch_page("pages/three_way_admixture.py")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown("""
### 🧠 Notes

- Use *Closest Populations* to explore similarity  
- Use *2-Way Admixture* for best-fit ancestry  
- Use *Fixed Mixture* to test hypotheses  

⚠️ Results depend on dataset and parameters.
""")