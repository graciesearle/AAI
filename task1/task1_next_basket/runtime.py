import os
import pandas as pd
import tensorflow as tf
import joblib
import logging
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'quick_reorder_lstm.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'lstm_scaler.pkl')
RESEARCH_PROD_PKL = os.path.join(BASE_DIR, 'prod_features.pkl') # From Task1.ipynb

PROD_DATA_DIR = os.path.join(BASE_DIR, 'production_data')
INSTACART_DATA_DIR = os.path.join(BASE_DIR, 'instacart_data')

# Production Snapshots
USER_FEATURES_CSV = os.path.join(PROD_DATA_DIR, 'user_features.csv')
UP_FEATURES_CSV = os.path.join(PROD_DATA_DIR, 'user_product_features.csv')
PROD_FEATURES_CSV = os.path.join(PROD_DATA_DIR, 'prod_features.csv')

# Global cache for assets
_model = None
_scaler = None
_prod_features = None
_research_prod_features = None

def _load_assets():
    """Load model, scaler, and product names into memory."""
    global _model, _scaler, _prod_features, _research_prod_features
    
    try:
        if _model is None:
            _model = tf.keras.models.load_model(MODEL_PATH)
            
        if _scaler is None:
            _scaler = joblib.load(SCALER_PATH)
            
        if os.path.exists(PROD_FEATURES_CSV):
            _prod_features = pd.read_csv(PROD_FEATURES_CSV)
            
        if os.path.exists(RESEARCH_PROD_PKL):
            logger.info("Loading Research Product Features (Parity Mode)...")
            _research_prod_features = pd.read_pickle(RESEARCH_PROD_PKL)
        
        return True
    except Exception as e:
        logger.error(f"Failed to load assets: {e}")
        return False

def predict_next_basket(user_id=None, top_n=5, demo_mode=False):
    """Main entry point for Next Basket Prediction."""
    global _model, _scaler, _prod_features, _research_prod_features
    
    if not _load_assets():
        return {"error": "Model assets not available"}

    # --- 1. DATA PREPARATION ---
    if demo_mode:
        # Default to a known research user (ID 1)
        if not user_id or int(user_id) < 10: 
            user_id = 1
        else:
            user_id = int(user_id)

        try:
            orders_path = os.path.join(INSTACART_DATA_DIR, 'orders.csv')
            prior_path = os.path.join(INSTACART_DATA_DIR, 'order_products__prior.csv')
            
            # Exact Parity: Match notebook feature engineering logic
            orders_df = pd.read_csv(orders_path)
            user_orders = orders_df[orders_df['user_id'] == user_id]
            
            # Notebook Feature: User Stats
            user_data_df = pd.DataFrame([{
                'user_id': user_id,
                'user_total_orders': user_orders['order_number'].max(),
                'user_avg_days_between': user_orders['days_since_prior_order'].mean()
            }])

            # Notebook Feature: User-Product Stats (Using chunking for memory safety)
            user_order_ids = set(user_orders['order_id'].tolist())
            relevant_priors = []
            for chunk in pd.read_csv(prior_path, chunksize=500000):
                matches = chunk[chunk['order_id'].isin(user_order_ids)]
                if not matches.empty:
                    relevant_priors.append(matches)
            
            prior_df = pd.concat(relevant_priors)
            
            # Exact Logic from Notebook cell 3
            user_prod_data = prior_df.groupby('product_id').size().reset_index(name='up_total_bought')
            user_prod_data['user_id'] = user_id
            
            # Exact Recency Logic from Notebook cell 5
            merged_prior = prior_df.merge(user_orders[['order_id', 'order_number']], on='order_id')
            recency = merged_prior.groupby('product_id')['order_number'].max().reset_index(name='up_last_order_num')
            user_prod_data = user_prod_data.merge(recency, on='product_id')

            # Use Research stats for exact parity
            if _research_prod_features is not None:
                stats_df = _research_prod_features
            else:
                return {"error": "Research 'prod_features.pkl' not found for exact parity."}

        except Exception as e:
            return {"error": f"Exact Parity logic failed: {str(e)}"}
    else:
        # PRODUCTION MODE
        if not os.path.exists(USER_FEATURES_CSV) or not os.path.exists(UP_FEATURES_CSV):
            return {"error": "Feature snapshots missing. Please run 'Export Next Basket Features'."}

        user_df = pd.read_csv(USER_FEATURES_CSV)
        up_df = pd.read_csv(UP_FEATURES_CSV)
        user_data_df = user_df[user_df['user_id'] == int(user_id)]
        user_prod_data = up_df[up_df['user_id'] == int(user_id)]
        stats_df = _prod_features

    # --- 2. MERGE & FEATURE ENGINEERING ---
    # Exact parity means merging on exactly the same columns as the notebook
    df = user_prod_data.merge(user_data_df, on='user_id')
    df = df.merge(stats_df, on='product_id', how='left')
    
    # Exact orders_since_last_purchase calculation
    df['orders_since_last_purchase'] = df['user_total_orders'] - df['up_last_order_num']
    df = df.fillna(0)

    # --- 3. INFERENCE ---
    feature_cols = [
        'up_total_bought', 'user_total_orders', 'user_avg_days_between',
        'prod_total_purchases', 'prod_reorder_rate', 'orders_since_last_purchase'
    ]
    
    X_scaled = _scaler.transform(df[feature_cols])
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    probs = _model.predict(X_lstm).ravel()
    df['score'] = probs

    # --- 4. FORMAT OUTPUT ---
    recommendations = df.sort_values('score', ascending=False).head(top_n)
    
    result = []
    # Try to find a products.csv to get the real names (e.g. 'Banana')
    prod_names = {}
    name_file = os.path.join(INSTACART_DATA_DIR, 'products.csv')
    if demo_mode and os.path.exists(name_file):
        n_df = pd.read_csv(name_file)
        prod_names = dict(zip(n_df['product_id'], n_df['product_name']))

    for _, row in recommendations.iterrows():
        p_id = int(row['product_id'])
        display_name = prod_names.get(p_id, f"Product {p_id}")
        
        result.append({
            "product_id": p_id,
            "product_name": display_name,
            "confidence": float(row['score']),
            "reorder_probability": "High" if row['score'] > 0.7 else "Medium" if row['score'] > 0.4 else "Low"
        })
    
    return result
