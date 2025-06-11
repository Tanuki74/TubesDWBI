import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Fungsi untuk memuat data (sesuaikan path file dengan lokasi data Anda)
@st.cache_data
def load_data():
    dim_customer = pd.read_csv("../data/olist_customers_dataset.csv")
    dim_order = pd.read_csv("../data/dim_order_cleaned.csv")
    dim_product = pd.read_csv("../data/dim_product_cleaned.csv")
    dim_order_item = pd.read_csv("../data/olist_order_items_dataset.csv")
    dim_order_payment = pd.read_csv("../data/olist_order_payments_dataset.csv")
    dim_seller = pd.read_csv("../data/olist_sellers_dataset.csv")
    fact_sales = pd.read_excel("../data/Fact_Sales.xls") 

    # Lakukan preprocessing data seperti di notebook Anda
    dim_order.dropna(axis=0, inplace=True)
    dim_product.dropna(axis=0, inplace=True)

    return dim_customer, dim_order, dim_product, dim_order_item, dim_order_payment, dim_seller, fact_sales

# Muat data
dim_customer, dim_order, dim_product, dim_order_item, dim_order_payment, dim_seller, fact_sales = load_data()

# --- Dashboard ---
st.title("Dashboard Analisis Penjualan Olist")

st.sidebar.header("Filter")

# selected_status = st.sidebar.multiselect(
#     "Pilih Status Pesanan",
#     dim_order['order_status'].unique(),
#     dim_order['order_status'].unique()
# )
# filtered_orders = dim_order[dim_order['order_status'].isin(selected_status)]

# --- KPI 1: Total Pendapatan ---
st.header("KPI 1: Total Pendapatan")
revenue_by_order = fact_sales.groupby('order_id')['price'].sum().reset_index()
total_revenue = revenue_by_order['price'].sum()
st.metric("Total Pendapatan Keseluruhan", f"Rp {total_revenue:,.2f}")

# --- KPI 2: Jumlah Pesanan ---
st.header("KPI 2: Jumlah Pesanan")
jumlah_pesanan = dim_order['order_id'].nunique()
st.metric("Jumlah Pesanan", jumlah_pesanan)

# --- KPI 3: Nilai Rata-rata Pesanan ---
st.header("KPI 3: Nilai Rata-rata Pesanan (AOV)")
average_order_value = revenue_by_order['price'].mean()
st.metric("Nilai Rata-rata Pesanan", f"Rp {average_order_value:,.2f}")

# Visualisasi AOV vs Prediksi (dari analisis regresi sebelumnya)
st.subheader("Visualisasi Prediksi AOV vs Aktual")
order_revenue_df = pd.merge(dim_order, revenue_by_order, on='order_id', how='inner')
features = ['order_status']
target = 'price'
regression_df = order_revenue_df.dropna(subset=features + [target])
X = regression_df[features]
y = regression_df[target]
X = pd.get_dummies(X, columns=features, drop_first=True)
# Pastikan X dan y tidak kosong sebelum membagi data
if not X.empty and not y.empty:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if not X_train.empty and not y_train.empty:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.set_title('Prediksi AOV vs Aktual')
        ax.set_xlabel('AOV Aktual')
        ax.set_ylabel('AOV Prediksi')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        st.pyplot(fig)
    else:
        st.write("Tidak cukup data untuk analisis regresi.")
else:
    st.write("Tidak cukup data untuk analisis regresi.")


# --- KPI 4: Jumlah Pelanggan Unik & Analisis RFM ---
st.header("KPI 4: Jumlah Pelanggan Unik & Analisis RFM")
jumlah_pelanggan_unik = dim_customer['customer_unique_id'].nunique()
st.metric("Jumlah Pelanggan Unik", jumlah_pelanggan_unik)

st.subheader("Analisis RFM (Recency, Frequency, Monetary)")

# Hitung RFM (seperti di notebook Anda)
customer_order_item_df = pd.merge(dim_order, dim_order_item, on='order_id')
customer_order_item_df = pd.merge(customer_order_item_df, dim_customer, on='customer_id')
customer_order_item_df['order_purchase_timestamp'] = pd.to_datetime(customer_order_item_df['order_purchase_timestamp'])
latest_date = customer_order_item_df['order_purchase_timestamp'].max()

rfm_df = customer_order_item_df.groupby('customer_unique_id').agg(
    Recency=('order_purchase_timestamp', lambda date: (latest_date - date.max()).days),
    Frequency=('order_id', 'nunique'),
    Monetary=('price', 'sum')
).reset_index()

st.write("Data RFM:")
st.dataframe(rfm_df.head())

# Clustering RFM (seperti di notebook Anda)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

# Gunakan K optimal yang telah Anda tentukan dari analisis sebelumnya
optimal_k = 3 # Ganti dengan K optimal yang Anda temukan

if len(rfm_scaled) > optimal_k: # Pastikan ada cukup data untuk clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    st.subheader(f"Hasil Clustering RFM (dengan K = {optimal_k})")
    cluster_analysis = rfm_df.groupby('Cluster').agg(
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Num_Customers=('customer_unique_id', 'count')
    ).reset_index()
    st.dataframe(cluster_analysis)

    # Visualisasi hasil clustering
    st.subheader("Visualisasi Clustering RFM")

    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=rfm_df, palette='viridis', s=100, alpha=0.6, ax=ax1)
    ax1.set_title('Customer Clustering (Recency vs Monetary)')
    ax1.set_xlabel('Recency (Days)')
    ax1.set_ylabel('Monetary Value')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='Frequency', y='Monetary', hue='Cluster', data=rfm_df, palette='viridis', s=100, alpha=0.6, ax=ax2)
    ax2.set_title('Customer Clustering (Frequency vs Monetary)')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Monetary Value')
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='Recency', y='Frequency', hue='Cluster', data=rfm_df, palette='viridis', s=100, alpha=0.6, ax=ax3)
    ax3.set_title('Customer Clustering (Recency vs Frequency)')
    ax3.set_xlabel('Recency (Days)')
    ax3.set_ylabel('Frequency')
    st.pyplot(fig3)
else:
    st.write("Tidak cukup data untuk melakukan clustering RFM.")


# --- KPI 5: Kategori Produk Terlaris ---
st.header("KPI 5: Kategori Produk Terlaris")

order_product_df = pd.merge(dim_order_item, dim_product, on='product_id', how='inner')
product_category_sales = order_product_df.groupby('product_category_name')['order_item_id'].count().reset_index()
product_category_sales.columns = ['product_category_name', 'number_of_sales']
product_category_sales_sorted = product_category_sales.sort_values(by='number_of_sales', ascending=False)

st.subheader("Top 10 Kategori Produk Terlaris (Berdasarkan Jumlah Penjualan)")
st.dataframe(product_category_sales_sorted.head(10))

# Visualisasi kategori produk terlaris
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='number_of_sales', y='product_category_name', data=product_category_sales_sorted.head(10), palette='viridis', ax=ax)
ax.set_title('Top 10 Most Selling Product Categories')
ax.set_xlabel('Number of Sales')
ax.set_ylabel('Product Category Name')
st.pyplot(fig)

st.subheader("Top 10 Kategori Produk Berdasarkan Pendapatan")
product_category_revenue = order_product_df.groupby('product_category_name')['price'].sum().reset_index()
product_category_revenue.columns = ['product_category_name', 'total_revenue']
product_category_revenue_sorted = product_category_revenue.sort_values(by='total_revenue', ascending=False)
st.dataframe(product_category_revenue_sorted.head(10))

# Visualisasi kategori produk dengan pendapatan terbesar
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='total_revenue', y='product_category_name', data=product_category_revenue_sorted.head(10), palette='magma', ax=ax)
ax.set_title('Top 10 Product Categories by Revenue')
ax.set_xlabel('Total Revenue')
ax.set_ylabel('Product Category Name')
st.pyplot(fig)