
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
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Crime Rate Prediction in India", layout="wide")

# 1. Title and Introduction
st.title("Crime Rate Analysis and Prediction in India")
st.markdown("""
Welcome to the Crime Rate Dashboard for India.  
This dashboard presents an insightful analysis of city-wise crime data, clusters high-risk areas using K-Means,  
visualizes the results on interactive maps, and concludes with a **prediction of crime rates for 2025**.
""")

# 2. Cluster Analysis
st.header("Clustering High-Risk Crime Zones using K-Means")
st.markdown("Cities are grouped into crime severity clusters using unsupervised machine learning.")
st.markdown("""
**What does this mean?**  
We grouped cities based on how serious the crime is in each place.  
It's like putting cities into 3 boxes:  
- **Red = High Crime**  
- **Orange = Medium Crime**  
- **Green = Low Crime**  
This helps decision-makers know which areas need the most attention.
""")
cluster_img_path = "crime_clusters_pca.png"
st.image(cluster_img_path, caption="K-Means Cluster Visualization", use_container_width=True)

# 3. Map Visualizations
st.header("Interactive Crime Maps")
st.markdown("The following maps help us visualize city crime intensity and patterns across India.")
st.markdown("""
**What do these maps show?**  
They make it easier to *see* where crime is high or low, using colors and map markers.  
- **Heatmap** shows how intense the crime is — the redder the area, the worse it is.  
- **Choropleth** colors entire states based on crime levels.  
- **Markers Map** lets you see exact crime locations across cities.
""")

col1, col2 , col3= st.columns(3)
with col1:
    st.markdown("**1. Crime Heatmap**")
    st.components.v1.html(open("heatmap.html", "r", encoding='utf-8').read(), height=500)

with col2:
    st.markdown("**2. Choropleth Map**")
    st.components.v1.html(open("choropleth.html", "r", encoding='utf-8').read(), height=500)

with col3:
    st.markdown("**3. Crime Markers Map**")
    st.components.v1.html(open("markers.html", "r", encoding='utf-8').read(), height=500)


# 4. Insights Section
st.header("Meaningful Insights & Victim Analysis")
st.markdown("""
**What does this section show?**  
This helps us understand *who* is most affected by crime — by age and gender.  
Knowing this helps build better protection systems and raise awareness for the most affected groups.
""")

insights_path = "victim_age_pca.png"
st.image(insights_path, caption="Victim Age Distribution", use_container_width=True)
st.markdown("This chart shows which age groups are most affected by crime. Are young people more at risk, or older ones?")

insights_path = "victim_gender_pca.png"
st.image(insights_path, caption="Victim Gender Distribution", use_container_width=True)
st.markdown("This chart shows whether more males or females are affected in different cities. This helps us understand who needs more protection in different areas.")

# 5. Linear Regression Crime Prediction
st.header("Crime Rate Prediction for 2025")
st.markdown("""
Using a method called **Linear Regression**, we made predictions about how crime numbers might look in the year 2025 for Indian cities.

This helps authorities and researchers prepare for the future and try to reduce risks ahead of time.
""")

# Load predictions
predicted_df = pd.read_csv("predicted_crimes_2025.csv")

# Display table
st.dataframe(predicted_df.sort_values(by="predicted_crime_2025", ascending=False))

# Visualize predictions
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(predicted_df['city'], predicted_df['predicted_crime_2025'], color="crimson")
plt.xticks(rotation=90)
plt.title("Predicted Crime Rates for 2025")
plt.ylabel("Crime Count")
st.pyplot(fig)

# 6. Conclusion
st.header("Conclusion")
st.markdown("""
This dashboard has walked you through the journey of understanding city-wise crime trends, cluster analysis,  
and future predictions based on current patterns.  

**Why it matters:**  
This can help policy makers, law enforcement, researchers, and even everyday citizens stay informed  
and focus attention on the cities and people who need it most.
""")

