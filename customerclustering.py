import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('customer_data.csv')
    return df

# Preprocess data
def preprocess_data(df):
    # Group by customer and aggregate
    customer_data = df.groupby('Customer').agg({
        'InvoiceAmount': 'sum',
        'InvoiceOutstandingBalance': 'sum',
        'DelayedPayment': lambda x: (x == 'Y').sum(),
        'DelinquentCustomer': lambda x: (x == 'Y').sum(),
        'CreditLimit': 'first',
        'CreditScore': 'first'
    }).reset_index()
    
    return customer_data

# Perform clustering
def perform_clustering(data):
    features = ['InvoiceAmount', 'InvoiceOutstandingBalance', 'DelayedPayment', 
                'DelinquentCustomer', 'CreditLimit', 'CreditScore']
    
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)
    
    return data

# Assign risk levels
def assign_risk_levels(data):
    cluster_risk = {
        0: 'Low Risk',
        1: 'Medium Risk',
        2: 'High Risk',
        3: 'No Risk'
    }
    data['RiskLevel'] = data['Cluster'].map(cluster_risk)
    return data

# Create visualization
def create_visualization(data):
    fig = px.scatter(data, x='CreditScore', y='InvoiceAmount', 
                     color='RiskLevel', hover_data=['Customer'],
                     color_discrete_map={'High Risk': 'red', 'Medium Risk': 'orange', 
                                         'Low Risk': 'green', 'No Risk': 'blue'})
    return fig

# Streamlit app
def main():
    st.title('Oracle ERP Customer Segmentation')
    
    df = load_data()
    customer_data = preprocess_data(df)
    clustered_data = perform_clustering(customer_data)
    risk_data = assign_risk_levels(clustered_data)
    
    st.subheader('Oracle ERP Customer Segmentation')
    fig = create_visualization(risk_data)
    st.plotly_chart(fig)
    
    st.subheader('Oracle ERP Customer Segmentation')
    st.dataframe(risk_data)

if __name__ == '__main__':
    main()