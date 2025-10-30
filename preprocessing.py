import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
df = pd.read_csv("adult.csv")

# 2. Clean column names and strip spaces
df.columns = df.columns.str.strip()
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# 3. Replace '?' with NaN and drop missing rows
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# 4. Separate features (X) and label (y)
X = df.drop('income', axis=1)   # Target column is 'income'
y = df['income']

# 5. Encode categorical columns using LabelEncoder
le = LabelEncoder()
for col in X.select_dtypes(include='object').columns:
    X[col] = le.fit_transform(X[col])

y = le.fit_transform(y)  # Encode target column

# 6. Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(" Data Preprocessing Completed!")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
