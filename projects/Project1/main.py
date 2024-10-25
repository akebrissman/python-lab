import json
import os
from enum import Enum

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model


class Classification(Enum):
    GOOD = 0
    MEDIUM = 1
    BAD = 2


class LocationData:
    def __init__(self):
        self.locations_df = None
        self.features_df = None
        self.target_df = None

        self.features_X_train = None
        self.features_X_test = None
        self.labels_y_train = None
        self.labels_y_test = None

        self.number_of_rows = 0
        self.number_of_features = 0
        self.number_of_output_categories = 0
        self.normalization_constant = 1
        self.scaler = MinMaxScaler()

    def is_loaded(self):
        return True if self.number_of_rows > 0 else False

    def load(self, file_name="location_training_data.csv"):
        try:
            # Load data from csv file
            current_directory = os.path.dirname(__file__)
            file_path = os.path.join(current_directory, file_name)
            self.locations_df = pd.read_csv(file_path, delimiter=';', decimal=',')

            # Drop blank lines
            self.locations_df.dropna(how="all", inplace=True)

            self.features_df = self.locations_df[["payed_parking", "apartments", "apartment_age"]]
            self.target_df = self.locations_df["rank"]

            # self.number_of_rows = len(self.features_df)
            # self.number_of_features = len(self.features_df.columns)
            self.number_of_rows = self.features_df.shape[0]
            self.number_of_features = self.features_df.shape[1]
            self.number_of_output_categories = len(Classification)  # GOOD(0), MEDIUM(1), BAD(2)

            print(f"Data laddat \nAntal datapunkter {len(self.locations_df)}")
        except Exception as e:
            print("Fel vid inläsning av data")
            print(e)

    def preprocess(self):
        try:
            # Koda kategoriska variabler
            # label_encoder = LabelEncoder()
            # self.features_df['sign'] = label_encoder.fit_transform(self.features_df['sign'])
            # self.target_df = label_encoder.fit_transform(self.target_df)

            #  Normalisera numeriska features
            self.features_df = self.scaler.fit_transform(self.features_df)
            # self.normalization_constant = np.max(features_X_train)
            # self.features_X_train = features_X_train  # / self.normalization_constant
            # self.features_X_test = features_X_test  # / self.normalization_constant

            # Dela upp data i 20% testdata, 80% träningsdata
            train_test_split_data = train_test_split(self.features_df, self.target_df, test_size=0.2, random_state=42)

            # X är features, y är labels
            features_X_train = train_test_split_data[0]  # Träningsdata
            features_X_test = train_test_split_data[1]  # Testdata
            labels_y_train = train_test_split_data[2]  # Tränings-labels
            labels_y_test = train_test_split_data[3]  # Test-labels

            # Konvertera etiketter till one-hot encoding
            # Nödvändigt för fler flerklass-klassificering
            # Kategori 0 blir[1, 0, 0]
            # Kategori 1 blir[0, 1, 0]
            # Kategori 2 blir[0, 0, 1]
            self.labels_y_train = tf.keras.utils.to_categorical(labels_y_train)
            self.labels_y_test = tf.keras.utils.to_categorical(labels_y_test)

            self.features_X_train = features_X_train
            self.features_X_test = features_X_test

            print("Data förberett för träning")
            print(f"Antal datapunkter träningsdata {len(features_X_train)}, Max-värde {np.max(features_X_train)} ")
            print(f"Antal datapunkter testdata {len(features_X_test)}, Max-värde {np.max(features_X_test)} ")
        except Exception as e:
            print("Fel vid förberedelse av data")
            print(e)


class UserData:
    def __init__(self):
        self.payed_parking = 0
        self.apartments = 0
        self.apartment_age = 0

    def get_user_input(self):
        while True:
            try:
                value = float(input(f"Parkeringsavgift: (0-10): "))
                if 0 <= value <= 100:
                    self.payed_parking = value
                    break
                else:
                    print("Ange ett nummer mellan 1 och 5.")
            except ValueError:
                print("Ange ett giltigt nummer!")

        while True:
            try:
                value = float(input(f"Antal lägenheter: (1-5): "))
                if 0 <= value <= 100:
                    self.apartments = value
                    break
                else:
                    print("Ange ett nummer mellan 1 och 5.")
            except ValueError:
                print("Ange ett giltigt nummer!")

        while True:
            try:
                value = float(input(f"Lägenheternas ålder: (1-5): "))
                if 0 <= value <= 100:
                    self.apartment_age = value
                    break
                else:
                    print("Ange ett nummer mellan 1 och 5.")

            except ValueError:
                print("Ange ett giltigt nummer!")


class ClassificationModel:
    def __init__(self, location_data=None):
        self.location_data = location_data
        self.model: tf.keras.models.Sequential = None
        self.history = None
        self.normalization_constant = location_data.normalization_constant if location_data else 1
        self.scaler = location_data.scaler if location_data else None
        self.is_trained = False

    def setup(self):
        # Skapar en enkel sekventiell modell, en linjär stack av lager för detta klassificerings-problem.
        # Grundläggande typ av neural network, varje lager kopplat till nästa
        # Input-Layer: Specificera input-shape = antal features för varje datapunkt, d.v.s antal kolumner
        # Hidden Layer 1: 10 noder per lager, bestäms efter experimentering
        # Hidden Layer 2: Fler lager, mer kapacitet att lära sig
        # Output Layer: Sista lagret har lika många noder som antalet target-klasser,
        # d.v.s modellen försöker välja mellan 3 kategorier för varje data-sample.

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.location_data.number_of_features,)),
            tf.keras.layers.Dense(10, activation='relu', ),
            tf.keras.layers.Dense(10, activation='relu', ),
            tf.keras.layers.Dense(self.location_data.number_of_output_categories, activation='softmax', ),
        ])

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        print("Model set up successfully.")

    def train(self, verbose=1):
        # Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch.
        """ Tolkning av verbose output:
            accuracy: Detta är träningsnoggrannheten för denna epoch. T ex 0.3421 (34.21%),
              vilket innebär att modellen korrekt klassificerade 34.21% av träningsdata.
            loss: Detta är förlustvärdet (loss) för modellen under träningen.
              Lägre värden indikerar bättre prestanda. T ex 2.8861 är relativt högt,
              vilket tyder på att modellen har svårt att göra korrekta förutsägelser.
            val_accuracy: Detta är noggrannheten på valideringsdatasetet. T ex 0.3000 (30%).
              Detta värde används för att utvärdera hur bra modellen generaliserar på osedd data.
            val_loss: Detta är förlustvärdet på valideringsdatasetet. T ex 2.7057, ju lägre, desto bättre.

            Riktvärden för att tolka resultaten:
            Accuracy: Generellt sett, om noggrannheten över 70% kan anses vara acceptabelt, beroende på problemet.
              För mer komplexa problem kan man behöva över 80%. I detta fall är 34.21% och 30% relativt låga.
            Loss: Ett förlustvärde under 1.0 är ofta en bra indikator,
              men det beror på vilken typ av problem du arbetar med. Ju lägre, desto bättre.
        """
        # self.history = self.model.fit(self.location_data.X_train,
        #                               self.location_data.y_train,
        #                               epochs=50,
        #                               validation_split=0.2,
        #                               verbose=verbose)

        early_stopping = EarlyStopping(
            monitor='val_loss',  # 'val_accuracy' eller 'val_loss'
            patience=10,  # antal epoker utan förbättring innan träningen stoppas
            verbose=1,  # ger output om tidigt stopp inträffar
            restore_best_weights=True  # återställer vikterna till den bästa modellen
        )

        target_accuracy = 0.80
        num_epochs = 50
        while True:
            # Dela upp träningsdatat i 80% träningsdata och 20% valideringsdata
            result = self.model.fit(self.location_data.features_X_train,
                                    self.location_data.labels_y_train,
                                    validation_split=0.2,
                                    epochs=num_epochs,
                                    callbacks=[early_stopping],
                                    verbose=verbose)

            validation_accuracy = result.history['val_accuracy'][-1]
            training_accuracy = result.history["accuracy"][-1]
            self.history = result.history

            print("")
            print(f"Acceptabel valideringsnoggrannhet {target_accuracy:.0%} \n"
                  f"Valideringsnoggrannhet {validation_accuracy:.2%} \n"
                  f"Träningsnoggrannhet {training_accuracy:.2%}  \n")

            # TODO: Skall man testa mot validation_accuracy eller training_accuracy?
            if validation_accuracy >= target_accuracy:
                self.is_trained = True
                break
            else:
                # Gör justeringar här, t ex, ändra hyperparametrar eller öka data
                # target_accuracy -= 0.10
                print("För dåligt, kör igen")

        print("Valideringsnoggrannhet (val_accuracy): Mått på hur bra modellen generaliserar till nya, osedda data.")
        print("Dvs hur många procent av valideringsdata (osedd data) som modellen klassificerar korrekt")
        print("")
        print("Träningsnoggrannhet (accuracy): Mått på hur bra modellen lär sig från träningsdata.")
        print("Dvs hur många procent av träningsdata som modellen klassificerade korrekt.")

        print(f"Model trained successfully.")

    def evaluation_graph(self):
        # Visualisera träningshistorik
        plt.figure(figsize=(12, 4))

        # Plot träningsnoggrannhet och valideringsnoggrannhet
        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], label='Träningsnoggrannhet')
        plt.plot(self.history['val_accuracy'], label='Valideringsnoggrannhet')
        plt.title('Modellens noggrannhet')
        plt.xlabel('Epoch')
        plt.ylabel('Noggrannhet')
        plt.legend()

        # Plot träningsförlust och valideringsförlust
        # Training Loss: Beräknas under varje epoch. Representerar hur bra model fitting.
        # Validation Loss: Beräknas på separat validerings-data. Visar eventuell model overfitting.
        # Om båda går nedåt under träning, lär sig modellen väl.
        # Om training loss går ner, men validation loss börjar gå upp, kan det indikera overfitting
        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], label='Träningsförlust')
        plt.plot(self.history['val_loss'], label='Valideringsförlust')
        plt.title('Modellens förlust')
        plt.xlabel('Epoch')
        plt.ylabel('Förlust')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def evaluate(self):
        # Utvärdera modellen med hjälp av testdata
        test_loss, test_acc = self.model.evaluate(self.location_data.features_X_test,
                                                  self.location_data.labels_y_test,
                                                  verbose=2)
        print("Modellens utvärdering genom testdata")
        print(f'Testnoggrannhet: {test_acc:.2%}')
        print(f'Testförlust {test_loss:.2%} dvs hur "säkra" var dessa förutsägelser.')
        if test_loss > .2:
            print("VARNING, Testförlusten indikerar låg säkerhet och potentiellt problematiska förutsägelser.")

        # Gör prediktioner på några exempel
        sample_predictions = self.model.predict(self.location_data.features_X_test[:3], verbose=1)
        print("\nSample predictions:")
        for i, prediction in enumerate(sample_predictions):
            print(f"Example {i + 1}: {prediction}")
            print(f"Predicted class: {np.argmax(prediction)}")
            print(f"Actual class: {np.argmax(self.location_data.labels_y_test[i])}")
            print(f"Result: {np.argmax(self.location_data.labels_y_test[i]) == np.argmax(prediction)}")
            print()

    def get_rank(self, user_preferences: UserData):
        data_to_predict = np.array(
            [[user_preferences.payed_parking, user_preferences.apartments, user_preferences.apartment_age]])
        # normalized_data = values / self.normalization_constant
        normalized_data = self.scaler.transform(data_to_predict)
        predictions = self.model.predict(normalized_data)
        return Classification(np.argmax(predictions))  # Index with the highest value is the enum value

    def save(self, file_name='model.h5'):
        file_type = file_name.split('.')[1]
        if file_type == "keras":
            # Spara modellen i keras-format
            self.model.save(file_name)
            # Spara scaler
            joblib.dump(self.scaler, 'minmax_scaler.pkl')
            # Spara normaliseringskonstanten
            with open('normalization_config.json', 'w') as f:
                json.dump({'normalization_constant': self.normalization_constant}, f)
            # Spara historiken
            with open('training_history.json', 'w') as f:
                json.dump(self.history, f)

        elif file_type == "h5":
            # Spara modellen i HDF5-format
            self.model.save(file_name)
            # Spara normaliseringskonstanten i samma HDF5-fil
            with h5py.File(file_name, 'a') as f:
                f.create_dataset('normalization_constant', data=self.normalization_constant)
        else:
            print("Unknown file format")

        print("Modellen sparad")

    def load(self, file_name="model.h5"):
        file_type = file_name.split('.')[1]
        if file_type == "keras":
            # Ladda modellen
            self.model = load_model(file_name)
            self.is_trained = True
            # Ladda scaler
            self.scaler = joblib.load('minmax_scaler.pkl')
            # Ladda normaliseringskonstanten
            with open('normalization_config.json', 'r') as f:
                config = json.load(f)
                self.normalization_constant = config['normalization_constant']
            # Ladda historiken
            with open('training_history.json', 'r') as f:
                self.history = json.load(f)

        elif file_type == "h5":
            # Ladda modellen
            self.model = load_model(file_name)
            # Ladda normaliseringskonstanten
            with h5py.File(file_name, 'r') as f:
                self.normalization_constant = f['normalization_constant'][()]
        else:
            print("Unknown file format")

        print("Modellen laddad")


class UserInterface:
    def __init__(self):
        self.location_data = LocationData()
        self.rank_model = ClassificationModel(self.location_data)
        self.user_data = UserData()

    def user_input(self):
        while True:
            self.user_data.get_user_input()
            rank = self.rank_model.get_rank(self.user_data)
            print(rank.name)
            exit_loop = input(f"\nDo you want to exit (y/n)")
            if exit_loop.lower() == "y":
                break

    def all_in_sequence(self):
        self.location_data.load()
        self.location_data.preprocess()

        self.rank_model.setup()
        self.rank_model.train(verbose=1)
        self.rank_model.save('rank_model.keras')

        self.rank_model.evaluate()

    def main_menu(self):
        print("Välkommen till Location Rank predictor!")

        while True:
            print("\n==============================")
            print("Ange funktion:")
            print("1. Ladda data för träning")
            print("2. Skapa modell") if self.location_data.is_loaded() else None
            print("3. Träna modell") if self.rank_model.model else None
            print("4. Utvärdera modell") if self.rank_model.is_trained and self.location_data.is_loaded() else None
            print("5. Visa graf") if self.rank_model.is_trained else None
            print("6. Spara modell") if self.rank_model.is_trained else None
            print("8. Ladda befintlig modell")
            print("9. Klassificera") if self.rank_model.is_trained else None
            print("0. Avsluta")

            choice = input("Välj ett alternativ: ")
            print("==============================\n")
            if choice == '1':
                self.location_data.load()
                self.location_data.preprocess()
            elif choice == '2':
                self.rank_model.setup()
            elif choice == '3':
                self.rank_model.train(verbose=1)
            elif choice == '4':
                self.rank_model.evaluate()
            elif choice == '5':
                self.rank_model.evaluation_graph()
            elif choice == '6':
                self.rank_model.save('rank_model.keras')
            elif choice == '8':
                self.rank_model.load('rank_model.keras')
            elif choice == '9':
                self.user_input()
            elif choice == 'all':
                self.all_in_sequence()
            elif choice == '0':
                print("Avslutar programmet.")
                break
            else:
                print("Ogiltigt alternativ, försök igen.")


if __name__ == "__main__":
    ui = UserInterface()
    ui.main_menu()
