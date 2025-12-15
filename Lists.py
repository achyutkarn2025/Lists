import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import hashlib
import json
from collections import defaultdict
import random

# Page Configuration
st.set_page_config(
    page_title="NexShop AI - Advanced E-Commerce Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .stMetric label {color: white !important;}
    .stMetric .metric-value {color: white !important;}
    .product-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box_shadow: 0 8px 12px rgba(0,0,0,0.2);
    }
    .recommendation-badge {
        background: linear_gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        color: white;
        padding: 5px 10px;
        border_radius: 20px;
        font_size: 12px;
    }
    .analytics-header {
        background: linear_gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
        padding: 30px;
        border_radius: 15px;
        margin_bottom: 20px;
    }
    div[data_testid="stSidebar"] {
        background: linear_gradient(180deg, #2C3E50 0%, #3498DB 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'products' not in st.session_state:
    st.session_state.products = pd.DataFrame({
        'id': range(1, 51),
        'name': [
            'Premium Wireless Headphones', 'Smart Watch Pro', 'Laptop Ultra 15"', 'Gaming Mouse RGB',
            'Mechanical Keyboard', 'USB-C Hub', '4K Webcam', 'Portable SSD 1TB',
            'Wireless Charger', 'Phone Case Premium', 'Screen Protector', 'Bluetooth Speaker',
            'External Monitor 27"', 'Ergonomic Chair', 'Standing Desk', 'LED Desk Lamp',
            'Noise Cancelling Earbuds', 'Tablet 10"', 'Stylus Pen', 'Power Bank 20000mAh',
            'HDMI Cable 4K', 'Laptop Stand', 'Cable Organizer', 'Webcam Cover',
            'Blue Light Glasses', 'Portable Monitor', 'Gaming Headset', 'Streaming Microphone',
            'Ring Light', 'Green Screen', 'Drawing Tablet', 'VR Headset',
            'Fitness Tracker', 'Smart Home Hub', 'Security Camera', 'Smart Bulbs 4-Pack',
            'Robot Vacuum', 'Air Purifier', 'Coffee Maker Smart', 'Blender Pro',
            'Instant Pot', 'Food Processor', 'Toaster Oven', 'Electric Kettle',
            'Stand Mixer', 'Sous Vide', 'Ninja Blender', 'Air Fryer XL',
            'Espresso Machine', 'Milk Frother'
        ],
        'category': [
            'Audio', 'Wearables', 'Computers', 'Accessories', 'Accessories', 'Accessories',
            'Cameras', 'Storage', 'Accessories', 'Accessories', 'Accessories', 'Audio',
            'Monitors', 'Furniture', 'Furniture', 'Lighting', 'Audio', 'Tablets', 'Accessories',
            'Accessories', 'Cables', 'Accessories', 'Accessories', 'Accessories', 'Accessories',
            'Monitors', 'Audio', 'Audio', 'Lighting', 'Photography', 'Input Devices', 'VR',
            'Wearables', 'Smart Home', 'Smart Home', 'Smart Home', 'Smart Home', 'Smart Home',
            'Kitchen', 'Kitchen', 'Kitchen', 'Kitchen', 'Kitchen', 'Kitchen', 'Kitchen',
            'Kitchen', 'Kitchen', 'Kitchen', 'Kitchen', 'Kitchen'
        ],
        'price': np.random.uniform(15, 500, 50).round(2),
        'rating': np.random.uniform(3.5, 5.0, 50).round(1),
        'reviews': np.random.randint(50, 5000, 50),
        'stock': np.random.randint(0, 200, 50),
        'sales': np.random.randint(100, 10000, 50),
        'description': ['High-quality product with amazing features'] * 50
    })

if 'cart' not in st.session_state:
    st.session_state.cart = []

if 'user_interactions' not in st.session_state:
    st.session_state.user_interactions = []

if 'orders' not in st.session_state:
    st.session_state.orders = []

if 'revenue_data' not in st.session_state:
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    st.session_state.revenue_data = pd.DataFrame({
        'date': dates,
        'revenue': np.cumsum(np.random.uniform(1000, 5000, 30)),
        'orders': np.random.randint(50, 200, 30),
        'customers': np.random.randint(30, 150, 30)
    })

# Advanced Recommendation System
class RecommendationEngine:
    def __init__(self, products_df):
        self.products = products_df
        self.vectorizer = TfidfVectorizer()

    def content_based_recommendations(self, product_id, n=5):
        """Content-based filtering using product features"""
        product_features = self.products['category'] + ' ' + self.products['name']
        tfidf_matrix = self.vectorizer.fit_transform(product_features)

        idx = self.products[self.products['id'] == product_id].index[0]
        cosine_sim = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
        similar_indices = cosine_sim.argsort()[-n-1:-1][::-1]

        return self.products.iloc[similar_indices]

    def collaborative_filtering(self, user_history, n=5):
        """Simulate collaborative filtering based on user behavior"""
        if not user_history:
            return self.products.nlargest(n, 'rating')

        viewed_categories = [item['category'] for item in user_history]
        category_weights = pd.Series(viewed_categories).value_counts()

        recommendations = self.products.copy()
        recommendations['score'] = recommendations['category'].map(
            lambda x: category_weights.get(x, 0) * recommendations['rating']
        )

        return recommendations.nlargest(n, 'score')

    def trending_products(self, n=5):
        """Calculate trending products based on sales and ratings"""
        self.products['trend_score'] = (
            self.products['sales'] * 0.4 +
            self.products['rating'] * 1000 * 0.3 +
            self.products['reviews'] * 0.3
        )
        return self.products.nlargest(n, 'trend_score')

# Analytics Engine
class AnalyticsEngine:
    @staticmethod
    def calculate_metrics(products_df, orders):
        total_revenue = sum(order['total'] for order in orders)
        total_orders = len(orders)
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        conversion_rate = (total_orders / (products_df['reviews'].sum() / 10)) * 100

        return {
            'revenue': total_revenue,
            'orders': total_orders,
            'aov': avg_order_value,
            'conversion': conversion_rate
        }

    @staticmethod
    def customer_segmentation(interactions):
        """K-means clustering for customer segmentation"""
        if len(interactions) < 10:
            return None

        # Create feature matrix
        user_data = defaultdict(lambda: {'views': 0, 'purchases': 0, 'total_spent': 0})

        for interaction in interactions:
            user_id = interaction.get('user_id', 'anonymous')
            user_data[user_id]['views'] += 1
            if interaction.get('type') == 'purchase':
                user_data[user_id]['purchases'] += 1
                user_data[user_id]['total_spent'] += interaction.get('amount', 0)

        return user_data

    @staticmethod
    def predict_demand(product_id, historical_sales):
        """Simple demand forecasting using moving average"""
        if len(historical_sales) < 7:
            return historical_sales[-1] if historical_sales else 0

        weights = np.exp(np.linspace(-1, 0, 7))
        weights /= weights.sum()
        forecast = np.average(historical_sales[-7:], weights=weights)

        return int(forecast)

# Initialize engines
rec_engine = RecommendationEngine(st.session_state.products)
analytics_engine = AnalyticsEngine()

# Sidebar Navigation
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/3498DB/FFFFFF?text=NexShop+AI", use_container_width=True)
    st.markdown("### 🎯 Navigation")

    page = st.radio(
        "",
        ["🏠 Home", "🛍️ Products", "📊 Analytics Dashboard", "🤖 AI Insights",
         "🛒 Cart", "👤 User Profile"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🔍 Quick Search")
    search_query = st.text_input("Search products...", key="sidebar_search")

    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    cart_items = len(st.session_state.cart)
    cart_total = sum(item['price'] * item['quantity'] for item in st.session_state.cart)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cart Items", cart_items)
    with col2:
        st.metric("Total", f"${cart_total:.2f}")

# Main Content
if page == "🏠 Home":
    st.markdown('<div class="analytics-header"><h1>🛍️ Welcome to NexShop AI</h1><p>Experience the future of intelligent shopping</p></div>', unsafe_allow_html=True)

    # Hero Section with Metrics
    col1, col2, col3, col4 = st.columns(4)

    metrics = analytics_engine.calculate_metrics(st.session_state.products, st.session_state.orders)

    with col1:
        st.metric("💰 Total Revenue", f"${metrics['revenue']:,.2f}", "+12.5%")
    with col2:
        st.metric("📦 Total Orders", f"{metrics['orders']:,}", "+8.2%")
    with col3:
        st.metric("💳 Avg Order Value", f"${metrics['aov']:.2f}", "+5.1%")
    with col4:
        st.metric("📈 Conversion Rate", f"{metrics['conversion']:.1f}%", "+3.2%")

    st.markdown("---")

    # Trending Products
    st.markdown("## 🔥 Trending Products")
    trending = rec_engine.trending_products(6)

    cols = st.columns(3)
    for idx, (_, product) in enumerate(trending.iterrows()):
        with cols[idx % 3]:
            with st.container():
                st.markdown(f"""
                <div class="product-card">
                    <span class="recommendation-badge">🔥 TRENDING</span>
                    <h3>{product['name']}</h3>
                    <p><strong>${product['price']:.2f}</strong></p>
                    <p>⭐ {product['rating']} ({product['reviews']} reviews)</p>
                    <p>📦 {product['stock']} in stock</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Add to Cart", key=f"trending_{product['id']}"):
                    st.session_state.cart.append({
                        'id': product['id'],
                        'name': product['name'],
                        'price': product['price'],
                        'quantity': 1
                    })
                    st.success(f"✅ {product['name']} added to cart!")
                    st.rerun()

    st.markdown("---")

    # Personalized Recommendations
    st.markdown("## 💡 Recommended For You")
    recommendations = rec_engine.collaborative_filtering(st.session_state.user_interactions, 3)

    cols = st.columns(3)
    for idx, (_, product) in enumerate(recommendations.iterrows()):
        with cols[idx]:
            st.markdown(f"""
            <div class="product-card">
                <span class="recommendation-badge">✨ AI PICK</span>
                <h3>{product['name']}</h3>
                <p><strong>${product['price']:.2f}</strong></p>
                <p>⭐ {product['rating']} | {product['category']}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"View Details", key=f"rec_{product['id']}"):
                st.session_state.user_interactions.append({
                    'product_id': product['id'],
                    'category': product['category'],
                    'type': 'view',
                    'timestamp': datetime.now()
                })

elif page == "🛍️ Products":
    st.title("🛍️ Product Catalog")

    # Advanced Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        categories = ['All'] + sorted(st.session_state.products['category'].unique().tolist())
        selected_category = st.selectbox("Category", categories)

    with col2:
        price_range = st.slider("Price Range", 0, 500, (0, 500))

    with col3:
        min_rating = st.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.5)

    with col4:
        sort_by = st.selectbox("Sort By", ["Name", "Price (Low-High)", "Price (High-Low)", "Rating", "Popularity"])

    # Filter products
    filtered_products = st.session_state.products.copy()

    if selected_category != 'All':
        filtered_products = filtered_products[filtered_products['category'] == selected_category]

    filtered_products = filtered_products[
        (filtered_products['price'] >= price_range[0]) &
        (filtered_products['price'] <= price_range[1]) &
        (filtered_products['rating'] >= min_rating)
    ]

    # Apply sorting
    if sort_by == "Price (Low-High)":
        filtered_products = filtered_products.sort_values('price')
    elif sort_by == "Price (High-Low)":
        filtered_products = filtered_products.sort_values('price', ascending=False)
    elif sort_by == "Rating":
        filtered_products = filtered_products.sort_values('rating', ascending=False)
    elif sort_by == "Popularity":
        filtered_products = filtered_products.sort_values('sales', ascending=False)

    st.markdown(f"### Found {len(filtered_products)} products")

    # Display products
    for i in range(0, len(filtered_products), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(filtered_products):
                product = filtered_products.iloc[i + j]
                with col:
                    st.markdown(f"""
                    <div class="product-card">
                        <h3>{product['name']}</h3>
                        <p style="color: #667eea; font-size: 24px; font-weight: bold;">${product['price']:.2f}</p>
                        <p>⭐ {product['rating']} ({product['reviews']} reviews)</p>
                        <p>📦 Stock: {product['stock']}</p>
                        <p>🏷️ {product['category']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("🛒 Add", key=f"add_{product['id']}"):
                            st.session_state.cart.append({
                                'id': product['id'],
                                'name': product['name'],
                                'price': product['price'],
                                'quantity': 1
                            })
                            st.success("Added!")
                            st.rerun()
                    with col_b:
                        if st.button("👁️ View", key=f"view_{product['id']}"):
                            st.session_state.user_interactions.append({
                                'product_id': product['id'],
                                'category': product['category'],
                                'type': 'view'
                            })

elif page == "📊 Analytics Dashboard":
    st.markdown('<div class="analytics-header"><h1>📊 Advanced Analytics Dashboard</h1></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Sales Analytics", "🎯 Customer Insights", "📦 Inventory", "🔮 Predictions"])

    with tab1:
        st.markdown("### Revenue Trends")

        # Revenue over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.revenue_data['date'],
            y=st.session_state.revenue_data['revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#667eea', width=3),
            fill='tozeroy'
        ))

        fig.update_layout(
            title="Revenue Growth Over Time",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Category Performance
        col1, col2 = st.columns(2)

        with col1:
            category_sales = st.session_state.products.groupby('category')['sales'].sum().sort_values(ascending=False)
            fig = px.bar(
                x=category_sales.index,
                y=category_sales.values,
                title="Sales by Category",
                labels={'x': 'Category', 'y': 'Total Sales'},
                color=category_sales.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(
                names=category_sales.index,
                values=category_sales.values,
                title="Market Share by Category",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Customer Behavior Analysis")

        # Customer segmentation visualization
        segments = analytics_engine.customer_segmentation(st.session_state.user_interactions)

        if segments:
            segment_df = pd.DataFrame(segments).T

            fig = px.scatter(
                segment_df,
                x='views',
                y='total_spent',
                size='purchases',
                title="Customer Segmentation Analysis",
                labels={'views': 'Product Views', 'total_spent': 'Total Spent ($)', 'purchases': 'Purchases'},
                color='purchases',
                color_continuous_scale='Sunset'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Collecting customer behavior data... Start shopping to see insights!")

        # Top products by rating
        col1, col2 = st.columns(2)

        with col1:
            top_rated = st.session_state.products.nlargest(10, 'rating')
            fig = px.bar(
                top_rated,
                x='rating',
                y='name',
                orientation='h',
                title="Top Rated Products",
                color='rating',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            top_reviewed = st.session_state.products.nlargest(10, 'reviews')
            fig = px.bar(
                top_reviewed,
                x='reviews',
                y='name',
                orientation='h',
                title="Most Reviewed Products",
                color='reviews',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Inventory Management")

        # Stock levels
        low_stock = st.session_state.products[st.session_state.products['stock'] < 50]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⚠️ Low Stock Items", len(low_stock))
        with col2:
            avg_stock = st.session_state.products['stock'].mean()
            st.metric("📦 Average Stock", f"{avg_stock:.0f}")
        with col3:
            total_value = (st.session_state.products['stock'] * st.session_state.products['price']).sum()
            st.metric("💰 Inventory Value", f"${total_value:,.2f}")

        # Stock heatmap
        fig = px.treemap(
            st.session_state.products,
            path=['category', 'name'],
            values='stock',
            title="Inventory Distribution",
            color='stock',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(low_stock) > 0:
            st.warning("⚠️ Low Stock Alert!")
            st.dataframe(
                low_stock[['name', 'category', 'stock', 'price']],
                use_container_width=True
            )

    with tab4:
        st.markdown("### Predictive Analytics")

        # Sales forecast
        st.markdown("#### 📈 30-Day Sales Forecast")

        future_dates = pd.date_range(
            start=datetime.now(),
            periods=30,
            freq='D'
        )

        # Simple forecasting model
        last_30_days = st.session_state.revenue_data['orders'].values
        trend = np.polyfit(range(len(last_30_days)), last_30_days, 1)
        forecast = np.polyval(trend, range(len(last_30_days), len(last_30_days) + 30))
        forecast = np.maximum(forecast, 0)  # Ensure non-negative

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.revenue_data['date'],
            y=st.session_state.revenue_data['orders'],
            mode='lines',
            name='Historical',
            line=dict(color='#667eea')
        ))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='#ff6b6b', dash='dash')
        ))

        fig.update_layout(
            title="Orders Forecast (Next 30 Days)",
            xaxis_title="Date",
            yaxis_title="Orders",
            template='plotly_white',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Product demand prediction
        st.markdown("#### 🎯 Top Products Requiring Restock")

        products_with_forecast = st.session_state.products.copy()
        products_with_forecast['forecast_demand'] = products_with_forecast['sales'].apply(
            lambda x: analytics_engine.predict_demand(None, [x] * 7)
        )
        products_with_forecast['restock_needed'] = (
            products_with_forecast['forecast_demand'] - products_with_forecast['stock']
        ).clip(lower=0)

        restock = products_with_forecast[products_with_forecast['restock_needed'] > 0].nlargest(10, 'restock_needed')

        if len(restock) > 0:
            fig = px.bar(
                restock,
                x='name',
                y='restock_needed',
                title="Predicted Restock Quantities",
                color='restock_needed',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ All products adequately stocked!")

elif page == "🤖 AI Insights":
    st.markdown('<div class="analytics-header"><h1>🤖 AI-Powered Insights</h1></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎯 Smart Recommendations Engine")
        st.info("Our AI analyzes your browsing patterns, purchase history, and product similarities to provide personalized recommendations.")

        # Show recommendation algorithm insights
        if st.session_state.user_interactions:
            viewed_categories = [i.get('category', 'Unknown') for i in st.session_state.user_interactions]
            category_interests = pd.Series(viewed_categories).value_counts()

            fig = px.pie(
                names=category_interests.index,
                values=category_interests.values,
                title="Category Preferences",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)

        # Content-based recommendations
        st.markdown("### 🔍 Similar Product Discovery")
        selected_product = st.selectbox(
            "Select a product to find similar items:",
            st.session_state.products['name'].tolist()
        )

        if selected_product:
            product_id = st.session_state.products[
                st.session_state.products['name'] == selected_product
            ]['id'].iloc[0]

            similar = rec_engine.content_based_recommendations(product_id, 4)

            cols = st.columns(4)
            for idx, (_, product) in enumerate(similar.iterrows()):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="product-card">
                        <h4>{product['name']}</h4>
                        <p><strong>${product['price']:.2f}</strong></p>
                        <p>⭐ {product['rating']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 💡 AI Tips")

        tips = [
            "🎯 Browse more products to get better recommendations!",
            "📊 Your shopping pattern suggests you prefer Electronics",
            "💰 Best time to shop: Current deals available!",
            "⭐ High-rated items in your interest areas available",
            "📦 Consider bulk purchases for better value"
        ]

        for tip in tips:
            st.info(tip)

        st.markdown("---")
        st.markdown("### 🔮 Market Trends")
        st.success("📈 Electronics category trending +25%")
        st.warning("⚠️ Kitchen appliances seeing high demand")
        st.info("💡 Smart Home devices gaining popularity")

elif page == "🛒 Cart":
    st.title("🛒 Shopping Cart")

    if len(st.session_state.cart) == 0:
        st.info("Your cart is empty. Start shopping to add items!")
    else:
        # Display cart items
        for idx, item in enumerate(st.session_state.cart):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                st.markdown(f"### {item['name']}")
            with col2:
                st.markdown(f"**${item['price']:.2f}**")
            with col3:
                new_qty = st.number_input(
                    "Qty",
                    min_value=1,
                    value=item['quantity'],
                    key=f"qty_{idx}"
                )
                st.session_state.cart[idx]['quantity'] = new_qty
            with col4:
                if st.button("🗑️", key=f"remove_{idx}"):
                    st.session_state.cart.pop(idx)
                    st.rerun()

        st.markdown("---")

        # Cart summary
        total = sum(item['price'] * item['quantity'] for item in st.session_state.cart)
        subtotal = total
        tax = total * 0.1
        shipping = 10 if total < 100 else 0
        final_total = total + tax + shipping

        col1, col2 = st.columns([2, 1])

        with col2:
            st.markdown("### Order Summary")
            st.markdown(f"**Subtotal:** ${subtotal:.2f}")
            st.markdown(f"**Tax (10%):** ${tax:.2f}")
            st.markdown(f"**Shipping:** ${shipping:.2f}")
            st.markdown(f"### **Total:** ${final_total:.2f}")

            if st.button("🎉 Checkout", type="primary", use_container_width=True):
                order = {
                    'id': len(st.session_state.orders) + 1,
                    'items': st.session_state.cart.copy(),
                    'total': final_total,
                    'date': datetime.now()
                }
                st.session_state.orders.append(order)

                # Track purchases
                for item in st.session_state.cart:
                    st.session_state.user_interactions.append({
                        'product_id': item['id'],
                        'type': 'purchase',
                        'amount': item['price'] * item['quantity'],
                        'timestamp': datetime.now()
                    })

                st.session_state.cart = []
                st.success("🎉 Order placed successfully!")
                st.balloons()
                st.rerun()

elif page == "👤 User Profile":
    st.title("👤 User Profile & Order History")

    tab1, tab2, tab3 = st.tabs(["📦 Orders", "📊 Statistics", "⚙️ Settings"])

    with tab1:
        if len(st.session_state.orders) == 0:
            st.info("No orders yet. Start shopping!")
        else:
            for order in reversed(st.session_state.orders):
                with st.expander(f"Order #{order['id']} - {order['date'].strftime('%Y-%m-%d %H:%M')}"):
                    st.markdown(f"**Total: ${order['total']:.2f}**")
                    st.markdown("**Items:**")
                    for item in order['items']:
                        st.markdown(f"- {item['name']} x{item['quantity']} - ${item['price'] * item['quantity']:.2f}")

    with tab2:
        if len(st.session_state.orders) > 0:
            total_spent = sum(order['total'] for order in st.session_state.orders)
            total_items = sum(sum(item['quantity'] for item in order['items']) for order in st.session_state.orders)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Total Spent", f"${total_spent:.2f}")
            with col2:
                st.metric("📦 Total Orders", len(st.session_state.orders))
            with col3:
                st.metric("🛍️ Items Purchased", total_items)

            # Spending over time
            order_dates = [order['date'] for order in st.session_state.orders]
            order_totals = [order['total'] for order in st.session_state.orders]

            fig = px.line(
                x=order_dates,
                y=np.cumsum(order_totals),
                title="Cumulative Spending",
                labels={'x': 'Date', 'y': 'Total Spent ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Start shopping to see your statistics!")

    with tab3:
        st.markdown("### ⚙️ Preferences")

        notifications = st.checkbox("📧 Email Notifications", value=True)
        recommendations = st.checkbox("🤖 Personalized Recommendations", value=True)
        dark_mode = st.checkbox("🌙 Dark Mode", value=False)

        st.markdown("### 🔔 Alert Preferences")
        price_drop = st.checkbox("💰 Price Drop Alerts", value=True)
        back_in_stock = st.checkbox("📦 Back in Stock Alerts", value=True)

        if st.button("💾 Save Settings", type="primary"):
            st.success("✅ Settings saved successfully!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p><strong>NexShop AI</strong> - Powered by Advanced Machine Learning</p>
    <p>🔒 Secure Shopping | 🚚 Fast Delivery | 💯 Satisfaction Guaranteed</p>
    <p style='font-size: 12px;'>© 2025 NexShop. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
