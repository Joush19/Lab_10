import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('spam text data.csv')

print("Primeras filas del dataset:")
print(df.head())

print("\n¿Existen valores NaN en el dataset?:")
print(df.isna().sum())

df['Category'] = df['Category'].map({'spam': 1, 'ham': 0})

vectorizer = CountVectorizer(stop_words='english') 
X = vectorizer.fit_transform(df['Message']) 
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = MultinomialNB()

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

precision = accuracy_score(y_test, y_pred)
print("\nPrecisión del modelo:", round(precision * 100, 2), "%")

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

predicciones_totales = modelo.predict(X)

df['Prediccion'] = ['spam' if pred == 1 else 'ham' for pred in predicciones_totales]

df.to_csv('spam_predictions_full.csv', index=False)

print("\nArchivo 'spam_predictions_full.csv' creado con las predicciones para cada mensaje.")
