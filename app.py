import os
from flask import Flask, request, jsonify, render_template # type: ignore
import joblib
import pandas as pd
from nlp_model import predict_category

try:
    import psycopg2
except ImportError:  # DB driver is optional in deploys without a database
    psycopg2 = None

# Product-history storage is optional: it only runs when DATABASE_URL is set.
# On a host without a database the save simply no-ops so recommendations
# never break. Locally, set DATABASE_URL=postgresql://postgres:root@localhost/eco_pack
DATABASE_URL = os.environ.get("DATABASE_URL")

def save_user_product(product_name, category, fragility):
    if not DATABASE_URL or psycopg2 is None:
        return  # no database configured — skip silently

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # normalize
        product_name = product_name.lower().strip()

        # check if exists
        cursor.execute("""
            SELECT 1 FROM user_products WHERE product_name = %s
        """, (product_name,))

        exists = cursor.fetchone()

        if not exists:
            cursor.execute("""
                INSERT INTO user_products (product_name, category, fragility)
                VALUES (%s, %s, %s)
            """, (product_name, category, fragility))

            conn.commit()

        cursor.close()
        conn.close()
    except Exception as e:  # never let a DB hiccup break the recommendation
        app.logger.warning("save_user_product skipped: %s", e)

app = Flask(__name__)

# LOAD DATASET (IMPORTANT 🔥)
df = pd.read_csv("ecopackai_materials_frozen.csv")

# Feature engineering
df['eco_score'] = (
    df['recyclability_percentage'] +
    df['biodegradability_score']
) / 2

df['cost_norm'] = (
    df['cost'] - df['cost'].min()
) / (df['cost'].max() - df['cost'].min())

df['co2_norm'] = (
    df['co2_score'] - df['co2_score'].min()
) / (df['co2_score'].max() - df['co2_score'].min())

model = joblib.load("model.pkl")

#ABOUT PAGE
@app.route('/')
def about():
    return render_template("about.html")

@app.route('/home')
def home():
    return render_template("index.html")

# FORM SUBMIT
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    product_name = data['product_name']
    product_name = product_name.lower().strip()
    product_category = predict_category(product_name)

    # simple fragility logic (keep for now)
    if product_category == "fragile":
        fragility = "high"
    elif product_category in ["electronics", "cosmetics"]:
        fragility = "medium"
    else:
        fragility = "low"
    save_user_product(product_name, product_category, fragility)
    shipping_type = data['shipping_type']
    sustainability_priority = data['sustainability_priority']

    filtered_df = df.copy()

    # Filtering logic (keep this)
    if fragility == "high":
        filtered_df = filtered_df[filtered_df['strength'] >= 3]

    if product_category == "food":
        filtered_df = filtered_df[filtered_df['biodegradability_score'] >= 7]

    # 🔥 IMPORTANT: Prepare features for model
    model_input = filtered_df[['strength', 'weight_capacity', 'eco_score', 'cost_norm', 'co2_norm']]

    # 🔥 AI PREDICTION
    filtered_df['predicted_score'] = model.predict(model_input)

    # OPTIONAL: adjust based on user preference
    if sustainability_priority == "high":
        filtered_df['predicted_score'] += filtered_df['eco_score'] * 0.5

    if shipping_type == "international":
        filtered_df['predicted_score'] -= filtered_df['co2_norm'] * 0.3

    # Sort results
    result = filtered_df.sort_values(by='predicted_score', ascending=False)

    results = result[['material_name', 'predicted_score']].head(5)

    # round properly
    results['predicted_score'] = results['predicted_score'].apply(lambda x: round(float(x), 2))

    results = results.to_dict(orient='records')

    return render_template("result.html",results=results,category=product_category,fragility=fragility)

def get_product_category(product_name):
    product_name = product_name.lower()

    if any(word in product_name for word in ["milk", "food", "fruit", "vegetable"]):
        return "food"
    
    elif any(word in product_name for word in ["phone", "laptop", "electronics", "tv"]):
        return "electronics"
    
    elif any(word in product_name for word in ["glass", "ceramic"]):
        return "fragile"
    
    else:
        return "general"
    
def analyze_product(product_name):
    product_name = product_name.lower()

    # CATEGORY
    if any(word in product_name for word in ["milk", "food", "fruit", "vegetable"]):
        category = "food"
    elif any(word in product_name for word in ["phone", "laptop", "tv", "electronics"]):
        category = "electronics"
    else:
        category = "general"

    # FRAGILITY
    if any(word in product_name for word in ["glass", "ceramic", "fragile"]):
        fragility = "high"
    elif any(word in product_name for word in ["laptop", "phone"]):
        fragility = "medium"
    else:
        fragility = "low"

    return category, fragility


if __name__ == "__main__":
    # Local dev server. In production the app is served by gunicorn (see Dockerfile).
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)