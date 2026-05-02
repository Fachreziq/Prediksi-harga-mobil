import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def main():
    df = pd.read_csv('data/car_data.csv')
    df.columns = df.columns.str.strip()

    # ===== ambil merk mobil =====
    car_names = df['Car_Name'].unique()
    joblib.dump(car_names, 'car_names.pkl')

    # ===== encoding =====
    le_fuel = LabelEncoder()
    le_trans = LabelEncoder()
    le_name = LabelEncoder()

    df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])
    df['Transmission'] = le_trans.fit_transform(df['Transmission'])
    df['Car_Name'] = le_name.fit_transform(df['Car_Name'])

    joblib.dump(le_name, 'encoder_name.pkl')

    # ===== fitur =====
    X = df[['Year', 'Kms_Driven', 'Fuel_Type', 'Transmission', 'Car_Name']]
    y = df['Selling_Price']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # ===== model =====
    model = Sequential([
        Dense(16, activation='relu', input_dim=5),
        Dense(8, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=10,
        validation_data=(X_test, y_test)
    )

    # ===== evaluasi =====
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)

    # ===== grafik =====
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Loss Training")
    plt.savefig('static/loss.png')
    plt.close()

    # ===== simpan =====
    model.save('model.h5')
    joblib.dump(scaler, 'scaler.pkl')

    print("✅ Model siap!")


if __name__ == "__main__":
    main()