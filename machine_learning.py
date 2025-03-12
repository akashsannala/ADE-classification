from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import pandas as pd

# Load dataset
X = df["text"]  # Features (text)
y = df["label"]  # Labels (0 or 1)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for efficiency
X_tfidf = vectorizer.fit_transform(X)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# Convert back to a DataFrame
df_balanced = pd.DataFrame(X_resampled.toarray(), columns=vectorizer.get_feature_names_out())
df_balanced["label"] = y_resampled

print(df_balanced["label"].value_counts())  # Check class balance



#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate Model Performance
print("Accuracy:", accuracy_score(y_test, y_pred))





from sklearn.svm import SVC

# Train SVM Model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate Model Performance
print("Accuracy:", accuracy_score(y_test, y_pred))




from xgboost import XGBClassifier

# Train XGBoost Model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate Model Performance
print("Accuracy:", accuracy_score(y_test, y_pred))



from sklearn.neural_network import MLPClassifier

# Train MLP Model
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate Model Performance
print("Accuracy:", accuracy_score(y_test, y_pred))


