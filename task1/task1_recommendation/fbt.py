import os
import logging
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

logger = logging.getLogger(__name__)

# Use absolute path relative to this file to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Groceries_dataset.csv")

_rules_cache = None


def _load_orders_from_db():
    """
    Placeholder for database loading. 
    In a decoupled architecture, AAI should fetch this data via a DESD API 
    endpoint or a shared export file rather than importing Django models directly.
    """
    logger.warning("Direct DB access from AAI is disabled. Falling back to CSV.")
    return pd.DataFrame(columns=["customer_id", "date", "item"])


def _load_orders_from_csv(csv_path=CSV_PATH):
    logger.info(f"Loading order history from CSV fallback: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = ["customer_id", "date", "item"]
    df.dropna(inplace=True)
    df["item"] = df["item"].str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    return df


def _build_rules(df, min_support=0.002, min_confidence=0.1):
    # Group into baskets and filter single-item baskets
    baskets = (
        df.groupby(["customer_id", "date"])["item"]
        .apply(list)
        .reset_index()
    )
    baskets.columns = ["customer_id", "date", "items"]
    baskets = baskets[baskets["items"].apply(len) >= 2].reset_index(drop=True)

    logger.info(f"Building rules from {len(baskets)} multi-item baskets...")

    if len(baskets) < 10:
        logger.warning("Not enough multi-item baskets to mine rules.")
        return None

    # One-hot encode
    te = TransactionEncoder()
    te_array = te.fit_transform(baskets["items"])
    encoded = pd.DataFrame(te_array, columns=te.columns_)

    # FP-Growth
    frequent_itemsets = fpgrowth(encoded, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        logger.warning(f"No frequent itemsets found at min_support={min_support}.")
        return None

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Keep only positive associations (lift > 1)
    rules = rules[rules["lift"] > 1.0].sort_values("lift", ascending=False)

    logger.info(f"Generated {len(rules)} association rules.")
    return rules


def get_rules(use_db=True, csv_path=CSV_PATH):
    global _rules_cache

    if _rules_cache is not None:
        return _rules_cache

    if use_db:
        try:
            df = _load_orders_from_db()
            rules = _build_rules(df, min_support=0.01, min_confidence=0.1)
            if rules is not None and len(rules) > 0:
                _rules_cache = rules
                return _rules_cache
            logger.warning("DB data insufficient — falling back to CSV.")
        except Exception as e:
            logger.error(f"DB load failed: {e} — falling back to CSV.")

    # CSV fallback with tuned parameters for sparse proxy data
    df = _load_orders_from_csv(csv_path)
    rules = _build_rules(df, min_support=0.001, min_confidence=0.1)
    _rules_cache = rules
    return _rules_cache


def recommend(basket, top_n=5, use_db=True, csv_path=CSV_PATH):
    rules = get_rules(use_db=use_db, csv_path=csv_path)

    if rules is None or rules.empty:
        logger.warning("No rules available — cannot make recommendations.")
        return []

    basket_set = set(item.lower().strip() for item in basket)
    recommendations = []

    for _, row in rules.iterrows():
        antecedents = set(row["antecedents"])
        consequents = set(row["consequents"])

        # Partial match: any overlap between antecedents and basket triggers rule
        if len(antecedents & basket_set) > 0:
            for item in consequents:
                if item not in basket_set:
                    recommendations.append({
                        "item": item,
                        "because_you_bought": list(antecedents),
                        "confidence": round(float(row["confidence"]), 3),
                        "lift": round(float(row["lift"]), 3),
                    })

    if not recommendations:
        return []

    # Deduplicate: keep highest lift per recommended item
    seen = {}
    for rec in recommendations:
        item = rec["item"]
        if item not in seen or rec["lift"] > seen[item]["lift"]:
            seen[item] = rec

    return sorted(seen.values(), key=lambda x: x["lift"], reverse=True)[:top_n]


def invalidate_cache():
    global _rules_cache
    _rules_cache = None
    logger.info("Recommendation rules cache cleared.")