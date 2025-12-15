import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="ShopEasy", page_icon="🛒", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .product-box {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: white;
    }
    .price {
        color: #B12704;
        font-size: 24px;
        font-weight: bold;
    }
    .rating {
        color: #FFA41C;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cart' not in st.session_state:
    st.session_state.cart = []

if 'products' not in st.session_state:
    st.session_state.products = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'name': [
            'Wireless Headphones',
            'Smart Watch',
            'Laptop Backpack',
            'USB-C Cable',
            'Phone Case',
            'Bluetooth Speaker',
            'Wireless Mouse',
            'Keyboard',
            'Water Bottle',
            'Power Bank',
            'Desk Lamp',
            'Notebook Set'
        ],
        'price': [49.99, 199.99, 35.99, 12.99, 15.99, 79.99, 25.99, 45.99, 18.99, 29.99, 32.99, 14.99],
        'rating': [4.5, 4.8, 4.3, 4.6, 4.2, 4.7, 4.4, 4.5, 4.1, 4.6, 4.3, 4.0],
        'category': ['Electronics', 'Electronics', 'Accessories', 'Electronics', 'Accessories', 
                    'Electronics', 'Electronics', 'Electronics', 'Home', 'Electronics', 'Home', 'Stationery'],
        'stock': [50, 30, 100, 200, 150, 40, 80, 60, 120, 70, 45, 90]
    })

# Sidebar
with st.sidebar:
    st.title("🛒 ShopEasy")
    st.markdown("---")
    
    page = st.radio("Menu", ["🏠 Home", "🛍️ Products", "🛒 Cart", "📦 Orders"])
    
    st.markdown("---")
    st.markdown("### Your Cart")
    cart_count = len(st.session_state.cart)
    cart_total = sum(item['price'] * item['qty'] for item in st.session_state.cart)
    st.write(f"Items: **{cart_count}**")
    st.write(f"Total: **${cart_total:.2f}**")

# HOME PAGE
if page == "🏠 Home":
    st.title("🛒 Welcome to ShopEasy!")
    st.subheader("Your one-stop online shop")
    
    st.markdown("---")
    
    # Featured Products
    st.markdown("## ⭐ Featured Products")
    
    featured = st.session_state.products.nlargest(3, 'rating')
    
    cols = st.columns(3)
    for idx, (_, product) in enumerate(featured.iterrows()):
        with cols[idx]:
            st.markdown(f"""
            <div class="product-box">
                <h3>{product['name']}</h3>
                <p class="price">${product['price']:.2f}</p>
                <p class="rating">⭐ {product['rating']}/5.0</p>
                <p>📦 Stock: {product['stock']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Add to Cart", key=f"home_{product['id']}"):
                st.session_state.cart.append({
                    'id': product['id'],
                    'name': product['name'],
                    'price': product['price'],
                    'qty': 1
                })
                st.success(f"✅ Added {product['name']} to cart!")
                st.rerun()
    
    st.markdown("---")
    
    # Categories
    st.markdown("## 📂 Shop by Category")
    
    categories = st.session_state.products['category'].unique()
    
    cols = st.columns(len(categories))
    for idx, category in enumerate(categories):
        with cols[idx]:
            count = len(st.session_state.products[st.session_state.products['category'] == category])
            st.info(f"**{category}**\n\n{count} items")

# PRODUCTS PAGE
elif page == "🛍️ Products":
    st.title("🛍️ All Products")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        categories = ['All'] + list(st.session_state.products['category'].unique())
        selected_category = st.selectbox("Category", categories)
    
    with col2:
        price_filter = st.selectbox("Price", ["All Prices", "Under $20", "$20-$50", "Over $50"])
    
    with col3:
        sort_option = st.selectbox("Sort by", ["Name", "Price: Low to High", "Price: High to Low", "Rating"])
    
    # Filter products
    filtered = st.session_state.products.copy()
    
    if selected_category != 'All':
        filtered = filtered[filtered['category'] == selected_category]
    
    if price_filter == "Under $20":
        filtered = filtered[filtered['price'] < 20]
    elif price_filter == "$20-$50":
        filtered = filtered[(filtered['price'] >= 20) & (filtered['price'] <= 50)]
    elif price_filter == "Over $50":
        filtered = filtered[filtered['price'] > 50]
    
    # Sort products
    if sort_option == "Price: Low to High":
        filtered = filtered.sort_values('price')
    elif sort_option == "Price: High to Low":
        filtered = filtered.sort_values('price', ascending=False)
    elif sort_option == "Rating":
        filtered = filtered.sort_values('rating', ascending=False)
    else:
        filtered = filtered.sort_values('name')
    
    st.markdown(f"### Showing {len(filtered)} products")
    st.markdown("---")
    
    # Display products
    for i in range(0, len(filtered), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(filtered):
                product = filtered.iloc[i + j]
                with cols[j]:
                    st.markdown(f"""
                    <div class="product-box">
                        <h3>{product['name']}</h3>
                        <p class="price">${product['price']:.2f}</p>
                        <p class="rating">⭐ {product['rating']}/5.0</p>
                        <p>Category: {product['category']}</p>
                        <p>📦 In Stock: {product['stock']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"🛒 Add to Cart", key=f"prod_{product['id']}"):
                        st.session_state.cart.append({
                            'id': product['id'],
                            'name': product['name'],
                            'price': product['price'],
                            'qty': 1
                        })
                        st.success("Added to cart!")
                        st.rerun()

# CART PAGE
elif page == "🛒 Cart":
    st.title("🛒 Shopping Cart")
    
    if len(st.session_state.cart) == 0:
        st.info("Your cart is empty!")
        st.write("Go to Products page to add items.")
    else:
        # Display cart items
        for idx, item in enumerate(st.session_state.cart):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**{item['name']}**")
            
            with col2:
                st.write(f"${item['price']:.2f}")
            
            with col3:
                new_qty = st.number_input("Qty", min_value=1, max_value=10, value=item['qty'], key=f"qty_{idx}")
                st.session_state.cart[idx]['qty'] = new_qty
            
            with col4:
                if st.button("❌", key=f"remove_{idx}"):
                    st.session_state.cart.pop(idx)
                    st.rerun()
            
            st.markdown("---")
        
        # Summary
        st.markdown("## 💳 Order Summary")
        
        subtotal = sum(item['price'] * item['qty'] for item in st.session_state.cart)
        tax = subtotal * 0.10
        shipping = 5.99 if subtotal < 50 else 0
        total = subtotal + tax + shipping
        
        col1, col2 = st.columns(2)
        
        with col2:
            st.write(f"**Subtotal:** ${subtotal:.2f}")
            st.write(f"**Tax (10%):** ${tax:.2f}")
            st.write(f"**Shipping:** ${shipping:.2f}")
            if shipping == 0:
                st.success("✅ Free shipping!")
            st.markdown(f"### **Total: ${total:.2f}**")
            
            st.markdown("---")
            
            if st.button("✅ Place Order", type="primary", use_container_width=True):
                if 'orders' not in st.session_state:
                    st.session_state.orders = []
                
                order = {
                    'order_id': len(st.session_state.orders) + 1,
                    'items': st.session_state.cart.copy(),
                    'total': total,
                    'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                st.session_state.orders.append(order)
                st.session_state.cart = []
                
                st.success("🎉 Order placed successfully!")
                st.balloons()
                st.rerun()

# ORDERS PAGE
elif page == "📦 Orders":
    st.title("📦 Your Orders")
    
    if 'orders' not in st.session_state or len(st.session_state.orders) == 0:
        st.info("You haven't placed any orders yet!")
    else:
        for order in reversed(st.session_state.orders):
            with st.expander(f"Order #{order['order_id']} - {order['date']} - ${order['total']:.2f}"):
                st.write("**Items:**")
                for item in order['items']:
                    st.write(f"- {item['name']} x {item['qty']} = ${item['price'] * item['qty']:.2f}")
                
                st.markdown("---")
                st.write(f"**Total: ${order['total']:.2f}**")
                st.success("✅ Delivered")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🛒 ShopEasy - Your Shopping Made Easy</p>
    <p>© 2025 ShopEasy. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
