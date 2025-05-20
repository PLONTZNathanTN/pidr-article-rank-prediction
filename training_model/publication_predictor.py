import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # For saving/loading the model
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns

# === PARAMÈTRES DE SÉLECTION PAR RANK ===
# Indiquez ici le nombre exact d’articles que vous voulez pour chaque rang
# Si la base n’en contient pas assez, on prendra tout ce qui existe.
SAMPLES_PER_RANK = {
    'A*': 268,
    'A': 268,
    'B': 268,
    'C': 268,
    'UnRanked': 268
}

# === MONGODB CONNECTION ===
URI = "mongodb+srv://nathan:pidr@pidr.jtdn1.mongodb.net/?retryWrites=true&w=majority&appName=PIDR"
client = MongoClient(URI)
db = client["ArticleDB"]
collection = db["Articles"]

# === FETCH DATA ===
BATCH_SIZE = sum(SAMPLES_PER_RANK.values())
articles_to_process = list(collection.find().limit(BATCH_SIZE * 5))  
# on prend un peu plus large pour pouvoir sampler par rang

if not articles_to_process:
    print("No articles found in the database.")
    exit()

# Convert to DataFrame
df = pd.DataFrame(articles_to_process)

# === CREATE TARGET VARIABLE ===
df['core_publication_rank'] = df['core_publication_rank'].fillna('UnRanked').replace('', 'UnRanked')
valid_ranks = list(SAMPLES_PER_RANK.keys())
df = df[df['core_publication_rank'].isin(valid_ranks)]

# === DOWNSAMPLE / UPSAMPLE PAR RANG ===
sampled_frames = []
for rank, n in SAMPLES_PER_RANK.items():
    group = df[df['core_publication_rank'] == rank]
    if len(group) >= n:
        sampled = group.sample(n, random_state=42)
    else:
        sampled = group  # s’il n’y en a pas assez, on prend tous
    sampled_frames.append(sampled)
df = pd.concat(sampled_frames).reset_index(drop=True)

y = df['core_publication_rank']
print("Target variable distribution after sampling:\n", y.value_counts())

# === PREPROCESSING ===
selected_features = [
    'abstract_size',
    'number_of_authors',
    'character',
    'page',
    'image',
    'version',
    'reference',
    'summary_gunning_fog_score',
    'hindex_average'
]
X = df[[col for col in selected_features if col in df.columns]]

numeric_features = []
for col in X.columns:
    if X[col].apply(lambda x: isinstance(x, (int, float))).all():
        numeric_features.append(col)
X = X[numeric_features]
print("Numeric features used for training:", X.columns.tolist())

# === TRAIN/TEST SPLIT ===
class_counts = y.value_counts()
if class_counts.min() < 2:
    print("Not enough samples for one of the classes. Skipping stratification.")
    stratify = None
else:
    stratify = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

print("Training and testing sets created.")

# === MODEL TRAINING ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Model trained.")

# === EVALUATION ===
y_pred = clf.predict(X_test)
labels_order = ['A*', 'A', 'B', 'C', 'UnRanked']
cm = confusion_matrix(y_test, y_pred, labels=labels_order)

print("Confusion Matrix (raw):\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === CONFUSION MATRIX VISUALIZATION ===
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order, yticklabels=labels_order)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - core_publication_rank")
plt.tight_layout()
plt.show()

# === FEATURE IMPORTANCE VISUALIZATION ===
importances = clf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()
