
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib


from preprocessing import X_train, X_test, y_train, y_test


model = DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=6)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(" Decision Tree Model Trained Successfully!")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 4. Save trained model for deployment
joblib.dump(model, 'model.pkl')
print("📁 Model saved as model.pkl (ready for deployment)")

