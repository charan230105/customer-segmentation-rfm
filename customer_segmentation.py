import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load datasets
# -----------------------------
orders = pd.read_csv("olist_orders_dataset.csv")
order_items = pd.read_csv("olist_order_items_dataset.csv")

# Convert date
orders['order_purchase_timestamp'] = pd.to_datetime(
    orders['order_purchase_timestamp']
)

# Merge datasets
df = orders.merge(order_items, on="order_id")

# Snapshot date
snapshot_date = df['order_purchase_timestamp'].max()

# -----------------------------
# RFM Calculation
# -----------------------------
recency = df.groupby('customer_id')['order_purchase_timestamp'].max()
recency = (snapshot_date - recency).dt.days

frequency = df.groupby('customer_id')['order_id'].count()
monetary = df.groupby('customer_id')['price'].sum()

rfm = pd.DataFrame({
    'Recency': recency,
    'Frequency': frequency,
    'Monetary': monetary
}).reset_index()

# -----------------------------
# Customer Segments
# -----------------------------
rfm['Segment'] = pd.cut(
    rfm['Monetary'],
    bins=[0,100,500,2000,15000],
    labels=['Low Value','Mid Value','High Value','Premium']
)

# -----------------------------
# RFM Scatter Plot
# -----------------------------
colors = {
    'Low Value':'blue',
    'Mid Value':'green',
    'High Value':'orange',
    'Premium':'red'
}

plt.figure(figsize=(8,5))

for seg in rfm['Segment'].unique():
    subset = rfm[rfm['Segment'] == seg]
    plt.scatter(
        subset['Frequency'],
        subset['Monetary'],
        label=seg,
        alpha=0.6,
        c=colors[seg]
    )

plt.legend()
plt.title("Customer Segmentation (Frequency vs Monetary)")
plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.tight_layout()
plt.savefig("rfm_scatter.png")
plt.show()

# -----------------------------
# Segment Revenue Chart
# -----------------------------
segment_revenue = rfm.groupby('Segment')['Monetary'].sum()

plt.figure(figsize=(8,5))
segment_revenue.plot(kind='bar')

plt.title("Revenue Contribution by Customer Segment")
plt.xlabel("Segment")
plt.ylabel("Total Revenue")

plt.tight_layout()
plt.savefig("segment_revenue.png")
plt.show()

# -----------------------
# Top 10 Customers 
# -----------------------
top_customers = rfm.sort_values(
    'Monetary', ascending=False
).head(10).reset_index(drop=True)

top_customers['Customer'] = [
    f"C{i+1}" for i in range(len(top_customers))
]

plt.figure(figsize=(10,5))

plt.bar(
    top_customers['Customer'],
    top_customers['Monetary']
)

plt.title("Top 10 High Value Customers")
plt.xlabel("Customer Rank")
plt.ylabel("Total Spend")

for i, v in enumerate(top_customers['Monetary']):
    plt.text(i, v + 100, f"{int(v)}", ha='center')

plt.tight_layout()
plt.savefig("top_customers.png")
plt.show()

print("Customer Segmentation Analysis Completed")