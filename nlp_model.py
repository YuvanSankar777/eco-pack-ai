from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔥 Training Data — kept as (item, category) pairs so the label always
# stays attached to its example. (Two parallel lists silently drift out of
# sync the moment one gets an extra entry.)
training_data = [
    # Food
    ("milk", "food"), ("bread", "food"), ("rice", "food"), ("apple", "food"),
    ("banana", "food"), ("vegetables", "food"), ("fruits", "food"),
    ("grocery", "food"), ("groceries", "food"), ("snacks", "food"),
    ("chocolate", "food"), ("biscuits", "food"), ("cookies", "food"),
    ("cereal", "food"), ("flour", "food"), ("sugar", "food"), ("salt", "food"),
    ("pasta", "food"), ("noodles", "food"), ("coffee", "food"), ("tea", "food"),
    ("eggs", "food"), ("cheese", "food"), ("butter", "food"), ("meat", "food"),
    ("fish", "food"), ("frozen food", "food"),

    # Electronics
    ("laptop", "electronics"), ("computer", "electronics"), ("phone", "electronics"),
    ("mobile", "electronics"), ("smartphone", "electronics"), ("tablet", "electronics"),
    ("charger", "electronics"), ("cable", "electronics"), ("adapter", "electronics"),
    ("tv", "electronics"), ("television", "electronics"), ("monitor", "electronics"),
    ("earphones", "electronics"), ("earpods", "electronics"), ("headphones", "electronics"),
    ("speaker", "electronics"), ("camera", "electronics"), ("keyboard", "electronics"),
    ("mouse", "electronics"), ("router", "electronics"), ("printer", "electronics"),
    ("battery", "electronics"), ("watch", "electronics"), ("console", "electronics"),
    ("drone", "electronics"),

    # Fragile
    ("glass", "fragile"), ("ceramic", "fragile"), ("mirror", "fragile"),
    ("cup", "fragile"), ("mug", "fragile"), ("plate", "fragile"),
    ("bowl", "fragile"), ("vase", "fragile"), ("bulb", "fragile"),
    ("light bulb", "fragile"), ("porcelain", "fragile"), ("china", "fragile"),
    ("crockery", "fragile"), ("crystal", "fragile"), ("figurine", "fragile"),
    ("ornament", "fragile"), ("picture frame", "fragile"), ("window", "fragile"),

    # Liquid
    ("oil", "liquid"), ("juice", "liquid"), ("water", "liquid"),
    ("bottle", "liquid"), ("liquid soap", "liquid"), ("shampoo", "liquid"),
    ("detergent", "liquid"), ("paint", "liquid"), ("sauce", "liquid"),
    ("syrup drink", "liquid"), ("beverage", "liquid"), ("wine", "liquid"),
    ("beer", "liquid"), ("ink", "liquid"), ("sanitizer", "liquid"),

    # Heavy
    ("machine", "heavy"), ("metal parts", "heavy"), ("equipment", "heavy"),
    ("tools", "heavy"), ("engine", "heavy"), ("motor", "heavy"),
    ("appliance", "heavy"), ("furniture", "heavy"), ("dumbbell", "heavy"),
    ("weights", "heavy"), ("generator", "heavy"), ("pump", "heavy"),
    ("gear", "heavy"), ("hardware", "heavy"),

    # Medical
    ("medicine", "medical"), ("tablets", "medical"), ("pills", "medical"),
    ("syrup", "medical"), ("medical kit", "medical"), ("first aid", "medical"),
    ("bandage", "medical"), ("syringe", "medical"), ("vaccine", "medical"),
    ("thermometer", "medical"), ("mask", "medical"), ("gloves", "medical"),
    ("ointment", "medical"), ("drugs", "medical"), ("capsules", "medical"),

    # Cosmetics
    ("cream", "cosmetics"), ("perfume", "cosmetics"), ("cosmetics", "cosmetics"),
    ("lotion", "cosmetics"), ("makeup", "cosmetics"), ("lipstick", "cosmetics"),
    ("foundation", "cosmetics"), ("mascara", "cosmetics"), ("moisturizer", "cosmetics"),
    ("sunscreen", "cosmetics"), ("serum", "cosmetics"), ("nail polish", "cosmetics"),
    ("deodorant", "cosmetics"), ("soap bar", "cosmetics"), ("face wash", "cosmetics"),
]

# Derive the parallel lists the rest of the module expects.
products = [item for item, _ in training_data]
categories = [category for _, category in training_data]

# 🔥 Vectorizer (converts text → numbers). Splitting each example into word
# tokens means a multi-word input like "wine glass bottle" can match the
# "glass" and "bottle" examples even though that exact phrase was never seen.
vectorizer = TfidfVectorizer()

# Train on existing product data
X = vectorizer.fit_transform(products)


# 🚀 MAIN FUNCTION (used in app.py)
def predict_category(product_name):
    product_name = product_name.lower().strip()

    # Convert input to vector
    input_vec = vectorizer.transform([product_name])

    # Compute similarity
    similarity = cosine_similarity(input_vec, X)

    # If the product isn't in the training vocabulary its vector is all
    # zeros, so every similarity is 0.0 and argmax() would blindly return
    # index 0 ("food"). Fall back to a neutral category instead of
    # mislabelling unknown products as food.
    if similarity.max() <= 0:
        return "general"

    # Get best match index
    index = similarity.argmax()

    # Return corresponding category
    return categories[index]