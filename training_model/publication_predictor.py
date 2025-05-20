import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_rank_classifier(collection, samples_per_rank=None):
    """
    Train a RandomForest classifier to predict core_publication_rank from MongoDB articles collection.

    Parameters:
    - collection: pymongo collection object
    - samples_per_rank: dict specifying how many samples per rank to use (default uses equal sampling)

    Returns:
    - clf: trained RandomForestClassifier model
    """
    if samples_per_rank is None:
        samples_per_rank = {
            'A*': 268,
            'A': 268,
            'B': 268,
            'C': 268,
            'UnRanked': 268
        }

    batch_size = sum(samples_per_rank.values())
    articles_to_process = list(collection.find().limit(batch_size * 5))

    if not articles_to_process:
        print("No articles found in the database.")
        return None

    df = pd.DataFrame(articles_to_process)

    # Prepare target variable
    df['core_publication_rank'] = df['core_publication_rank'].fillna('UnRanked').replace('', 'UnRanked')
    valid_ranks = list(samples_per_rank.keys())
    df = df[df['core_publication_rank'].isin(valid_ranks)]

    # Sample per rank
    sampled_frames = []
    for rank, n in samples_per_rank.items():
        group = df[df['core_publication_rank'] == rank]
        if len(group) >= n:
            sampled = group.sample(n, random_state=42)
        else:
            sampled = group
        sampled_frames.append(sampled)
    df = pd.concat(sampled_frames).reset_index(drop=True)

    y = df['core_publication_rank']
    print("Target variable distribution after sampling:\n", y.value_counts())

    # Select features
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

    numeric_features = [col for col in X.columns if X[col].apply(lambda x: isinstance(x, (int, float))).all()]
    X = X[numeric_features]
    print("Numeric features used for training:", X.columns.tolist())

    # Train/test split
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    print("Training and testing sets created.")

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    print("Model trained.")

    # Evaluate
    y_pred = clf.predict(X_test)
    labels_order = list(samples_per_rank.keys())
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)

    print("Confusion Matrix (raw):\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order, yticklabels=labels_order)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - core_publication_rank")
    plt.tight_layout()
    plt.show()

    # Plot feature importance
    importances = clf.feature_importances_
    feature_names = X.columns

    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.show()

    return clf
