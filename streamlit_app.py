
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Crime Rate Prediction in India", layout="wide")

# --- 1. Title and Introduction ---
st.title(" Crime Rate Analysis and Prediction Dashboard")
st.markdown("""
Welcome to the **Indian Crime Threat Intelligence Dashboard**. 
Here, we explore how crime is distributed across cities, identify high-risk zones using machine learning, 
and forecast possible 2025 crime rates to support smarter safety planning.
""")

# --- 2. Cluster Analysis ---
st.header("🔍 Cluster Analysis: Identifying Crime Zones")
st.markdown("""
Using **K-Means Clustering**, we grouped cities into risk zones based on crime characteristics. 
This allows security forces and policymakers to focus efforts based on the nature and intensity of crime.

**What the Graph Shows:**
- Each dot represents a city (positioned using PCA for clarity).
- Colors indicate different cluster types.
- Black 'X' marks the center of each cluster.

**Cluster Interpretations:**
| Color | Cluster Label | Description |
|-------|----------------|-------------|
| 🔴 Red | **Danger Zones** | High concentration of violent or armed crimes. |
| 🟠 Orange | **Risky Areas** | Widespread but moderate risk crime environments. |
| 🟢 Green | **Public Order Zones** | Areas with civil unrest or crowd-related threats. |
| 🟣 Purple | **Specific Threat Zones** | Cities with targeted, high-value or rare threats. |

> This clustering helps create **zone-specific policies** rather than generic safety responses.
""")
st.image("crime_clusters_pca.png", caption="K-Means Crime Clustering in PCA-Reduced Space", use_container_width=True)

# --- 3. Interactive Maps ---
st.header("🗺️ Crime Distribution Across India")
st.markdown("""
Visual maps make it easier to observe crime patterns geographically.
Choose your preferred map for more intuitive exploration:

- **Heatmap**: Highlights intensity using color gradients.
- **Choropleth**: Shades entire states based on crime.
- **Markers**: Pinpoint crimes city by city.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader(" Crime Heatmap")
    st.components.v1.html(open("heatmap.html", "r", encoding='utf-8').read(), height=500)

with col2:
    st.subheader(" Choropleth Map")
    st.components.v1.html(open("choropleth.html", "r", encoding='utf-8').read(), height=500)

with col3:
    st.subheader("📍 City-wise Markers")
    st.components.v1.html(open("markers.html", "r", encoding='utf-8').read(), height=500)

# --- 4. Victim Analysis ---
st.header(" Victim Demographics Analysis")
st.markdown("""
These charts break down **who** is most affected by crime:
- By **age**: Are youth or elderly more at risk?
- By **gender**: Which gender faces more frequent victimization?

Understanding this helps shape **targeted public awareness** and protection.
""")
st.image("victim_age_pca.png", caption="Victim Age Group Distribution", use_container_width=True)
st.image("victim_gender_pca.png", caption="Victim Gender Distribution", use_container_width=True)

# --- 4.5 Crime Domain Heatmap ---
st.header("📊 Crime Types by City")
st.markdown("""
This heatmap shows which **crime types** are more common in which cities.
Useful for understanding city-specific crime trends and resource allocation.
""")

# Load and preprocess data
df = pd.read_csv("crime_data_with_cities.csv")
domain_cols = [col for col in df.columns if col.startswith('crime_domain_')]
df_melted = df.melt(id_vars='city', value_vars=domain_cols, var_name='crime_domain', value_name='count')
df_melted['crime_domain'] = df_melted['crime_domain'].str.replace('crime_domain_', '').str.replace('_', ' ')
df_melted = df_melted[df_melted['count'] > 0]
heatmap_data = df_melted.groupby(['city', 'crime_domain']).size().unstack().fillna(0)

# Plot
fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(heatmap_data, cmap="YlOrBr", linewidths=0.5, annot=True, fmt='g', ax=ax)
ax.set_title("Crime Domain Distribution by City", fontsize=16)
ax.set_xlabel("Crime Domain")
ax.set_ylabel("City")
plt.xticks(rotation=45)
st.pyplot(fig)

# --- 5. Prediction Section ---
st.header("📈 Crime Rate Prediction for 2025")
st.markdown("""
Using **Linear Regression**, we predict the likely **crime count per city** in 2025.

This helps law enforcement plan ahead for crime prevention, budgeting, and capacity building.
""")

# Prediction data
predicted_df = pd.read_csv("predicted_crimes_2025.csv")
st.dataframe(predicted_df.sort_values(by="predicted_crime_2025", ascending=False))

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(predicted_df['city'], predicted_df['predicted_crime_2025'], color="crimson")
plt.xticks(rotation=90)
plt.title("Predicted Crime Rates for 2025")
plt.ylabel("Predicted Crime Count")
st.pyplot(fig)

# --- 6. Conclusion ---
st.header(" Conclusion: Smarter Crime Strategy")
st.markdown("""
This dashboard has provided a comprehensive exploration of crime patterns across Indian cities using data-driven techniques. By applying unsupervised clustering (K-Means), geospatial mapping, and predictive analytics, we’ve uncovered key insights into where crime is most concentrated, which populations are most vulnerable, and what trends might emerge in the near future.

Key Takeaways:

-Cluster analysis helped us classify urban zones by crime intensity, aiding law enforcement prioritization.

-Maps and visualizations made crime hotspots easily identifiable at a glance.

-Victim profiling shed light on the demographics most affected, guiding awareness and policy efforts.

-Predictive modeling gave a forward-looking view into crime trends for 2025.

Armed with these insights, decision-makers can make more informed, targeted, and proactive interventions to improve safety and reduce crime across the nation.""")
