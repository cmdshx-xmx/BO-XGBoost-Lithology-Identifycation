# Split the dataset into training and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_imputed.values,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Standardize the training and test sets
scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE only to the standarized training set
X_xgb, y_xgb = X_train_full_scaled, y_train_full

min_samples = np.bincount(y_xgb).min()
safe_k = max(1, min(5, min_samples - 1))

smote = SMOTE(
    random_state=42,
    k_neighbors=safe_k
)

X_xgb_sm, y_xgb_sm = smote.fit_resample(
    X_xgb,
    y_xgb
)