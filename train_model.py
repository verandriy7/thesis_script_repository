# Script per l'addestramento dei modelli RF, SVM e DNN

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop

# Caricamento del training set
df = pd.read_csv('train.csv')
X = df.drop(columns=['Label'])
y = df['Label']

# RANDOM FOREST
rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
rf.fit(X, y)
joblib.dump(rf, 'random_forest_model.pkl')

# SVM
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm.fit(X, y)
joblib.dump(svm, 'svm_model.pkl')

# DNN
dnn = Sequential()
dnn.add(Dense(128, activation='relu', input_dim=X.shape[1]))
dnn.add(Dropout(0.5))
dnn.add(Dense(64, activation='relu'))
dnn.add(Dense(1, activation='sigmoid'))
dnn.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
dnn.fit(X, y, epochs=20, batch_size=32, callbacks=[early_stop], verbose=1)
dnn.save('dnn_model.h5')
