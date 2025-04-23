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
