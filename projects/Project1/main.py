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


class QualityRating(Enum):
    EXCELLENT = 0
    GOOD = 1
    FAIR = 2
    POOR = 3
    BAD = 4


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
            self.locations_df.dropna(inplace=True)

            # För att förbättra träningen så kan vi duplicera datat
            self.locations_df = self.locations_df._append(self.locations_df, ignore_index=True)
            self.locations_df = self.locations_df._append(self.locations_df, ignore_index=True)

            self.features_df = self.locations_df[
                ["Invånare", "Andel lägenheter", "Andel BRF", "Lägenhetsålder", "Medianinkomst", "Områdestyp",
                 "Tidigt på plats", "Elbilspenetration", "Skyltning"]]
            self.target_df = self.locations_df["Rating"]

            # self.number_of_rows = len(self.features_df)
            # self.number_of_features = len(self.features_df.columns)
            self.number_of_rows = self.features_df.shape[0]
            self.number_of_features = self.features_df.shape[1]
            self.number_of_output_categories = len(QualityRating)  # EXCELLENT -- BAD

            if len(self.target_df.unique()) < self.number_of_output_categories:
                print(f"Det finns färre unika labels i datat än de {len(QualityRating)} som stöds")
                self.number_of_output_categories = len(self.target_df.unique())
            elif len(self.target_df.unique()) > self.number_of_output_categories:
                print(f"Varning, Det finns fler unika labels i datat än de {len(QualityRating)} som stöds")

            print(f"Data laddat \nAntal datapunkter {len(self.locations_df)}")
        except Exception as e:
            print("Fel vid inläsning av data")
            print(e)

    def preprocess(self):
        try:
            # Koda kategoriska variabler
            # Klassificeringarna blir numeriska värden tagna från vår enum QualityRating
            self.target_df = self.target_df.apply(lambda x: QualityRating[x].value)

            # TODO: Hur stor påverkan har det att minska stora värden t ex för Invånare?
            # För att minska påverkan från "Invånare" divideras den med ett stort tal, som 10 000, innan MinMaxScaler.
            # self.features_df['Invånare'] = self.features_df['Invånare'] / 10000
            # self.features_df.loc[:, 'Invånare'] = self.features_df['Invånare'] / 10000
            # self.features_df.loc[:, 'Medianinkomst'] = self.features_df['Medianinkomst'] / 10000
            # self.features_df['Invånare'] = self.self.features_df['Invånare'].apply(lambda x: x/1000)
            # self.features_df["Invånare"] = self.features_df.apply(lambda x: x["Invånare"]/1000)
            for i in range(len(self.features_df)):
                self.features_df['Invånare'].values[i] = self.features_df['Invånare'].values[i] / 10000

            # TODO: Vilken effekt har det att blanda features med olika typer av värden?.
            #   Medianinkomst 30000, Områdestyp (1, 2, eller 3), Andel lägenheter (procent 0 till 1)

            # Normalisera numeriska features
            # Alla numeriska värden skalas ner till mellan 0 och 1.
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
            # Kategori 0 blir [1, 0, 0]
            # Kategori 1 blir [0, 1, 0]
            # Kategori 2 blir [0, 0, 1] osv.
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
        self.people = 0
        self.apartments = 0
        self.brf = 0
        self.apartment_age = 0
        self.mean_income = 0
        self.area_type = 0
        self.early = 0
        self.ev_penetration = 0
        self.sign = 0

    def get_user_input(self):
        while True:
            try:
                value = float(input(f"Invånare: "))
                if 0 <= value <= 100000:
                    self.people = value
                    break
                else:
                    print("Ange ett nummer mellan 1 och 100000.")
            except ValueError:
                print("Ange ett giltigt nummer!")

        while True:
            try:
                value = float(input(f"Andel lägenheter %: (0-100): "))
                if 0 <= value <= 100:
                    self.apartments = value / 100
                    break
                else:
                    print("Ange ett nummer mellan 0 och 100.")
            except ValueError:
                print("Ange ett giltigt nummer!")

        while True:
            try:
                value = float(input(f"Andel BRF %: (0-100): "))
                if 0 <= value <= 100:
                    self.brf = value / 100
                    break
                else:
                    print("Ange ett nummer mellan 0 och 100.")
            except ValueError:
                print("Ange ett giltigt nummer!")

        while True:
            try:
                value = float(input(f"Lägenhetsålder: (0-100): "))
                if 0 <= value <= 100:
                    self.apartment_age = value
                    break
                else:
                    print("Ange ett nummer mellan 0 och 100.")

            except ValueError:
                print("Ange ett giltigt nummer!")

        while True:
            try:
                value = float(input(f"Medelinkomst: (0-1000000): "))
                if 0 <= value <= 1000000:
                    self.mean_income = value
                    break
                else:
                    print("Ange ett nummer mellan 0 och 1000000.")

            except ValueError:
                print("Ange ett giltigt nummer!")

        while True:
            try:
                value = float(input(f"Områdestyp: (0-5): "))
                if 0 <= value <= 5:
                    self.area_type = value
                    break
                else:
                    print("Ange ett nummer mellan 0 och 5.")

            except ValueError:
                print("Ange ett giltigt nummer!")

        while True:
            try:
                value = float(input(f"Områdestyp: (0-2): "))
                if 0 <= value <= 5:
                    self.early = value
                    break
                else:
                    print("Ange ett nummer mellan 0 och 2.")

            except ValueError:
                print("Ange ett giltigt nummer!")

        while True:
            try:
                value = float(input(f"EV pen %: (0-100): "))
                if 0 <= value <= 100:
                    self.ev_penetration = value / 100
                    break
                else:
                    print("Ange ett nummer mellan 0 och 100.")

            except ValueError:
                print("Ange ett giltigt nummer!")

        while True:
            try:
                value = float(input(f"Skyltning: (0-2): "))
                if 0 <= value <= 2:
                    self.sign = value
                    break
                else:
                    print("Ange ett nummer mellan 0 och 2.")

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

        # TODO: Hur vet jag hur många neuroner (units) jag skall använda på mina lager
        # Hur vet jag hur många lager jag skall ha?
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.location_data.number_of_features,)),
            tf.keras.layers.Dense(units=10, activation='relu', ),
            tf.keras.layers.Dense(units=10, activation='relu', ),
            tf.keras.layers.Dense(units=self.location_data.number_of_output_categories, activation='softmax', ),
        ])

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        print("Modellen Keras Sequential har skapats och kompilerats.")

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

        target_accuracy = 0.50
        num_epochs = 100
        loop = 0

        early_stopping = EarlyStopping(
            monitor='val_loss',  # 'val_accuracy' eller 'val_loss'
            patience=num_epochs / 2,  # antal epoker utan förbättring innan träningen stoppas
            verbose=1,  # ger output om tidigt stopp inträffar
            restore_best_weights=True  # återställer vikterna till den bästa modellen
        )

        # TODO: Vad är default för batch_size?
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
                if loop < 2:
                    print("För dåligt, kör igen")
                    loop += 1
                else:
                    self.is_trained = True
                    break

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

        # Gör prediktioner på de tre första raderna från test datat
        sample_predictions = self.model.predict(self.location_data.features_X_test[:3], verbose=1)
        # TODO: Skriv ut vilka labels som prediction gick bra respektive dålig för.
        print("\nSample predictions:")
        for i, prediction in enumerate(sample_predictions):
            print(f"Example {i + 1}: {prediction}")
            print(f"Predicted class: {np.argmax(prediction)}")
            print(f"Actual class: {np.argmax(self.location_data.labels_y_test[i])}")
            print(f"Result: {np.argmax(self.location_data.labels_y_test[i]) == np.argmax(prediction)}")
            print()

    def rate(self, user_preferences: UserData):
        # TODO: Skapa en array med alla värden från user_preferences
        # data_to_predict = np.array(
        #     [[user_preferences.people, user_preferences.apartments, user_preferences.brf,
        #       user_preferences.apartment_age, user_preferences.mean_income, user_preferences.area_type,
        #       user_preferences.early, user_preferences.ev_penetration, user_preferences.sign]])

        data_to_predict = pd.DataFrame()
        data_to_predict['Invånare'] = [user_preferences.people]
        data_to_predict['Andel lägenheter'] = [user_preferences.apartments]
        data_to_predict['Andel BRF'] = [user_preferences.brf]
        data_to_predict['Lägenhetsålder'] = [user_preferences.apartment_age]
        data_to_predict['Medianinkomst'] = [user_preferences.mean_income]
        data_to_predict['Områdestyp'] = [user_preferences.area_type]
        data_to_predict['Tidigt på plats'] = [user_preferences.early]
        data_to_predict['Elbilspenetration'] = [user_preferences.ev_penetration]
        data_to_predict['Skyltning'] = [user_preferences.sign]

        # normalized_data = values / self.normalization_constant
        normalized_data = self.scaler.transform(data_to_predict)
        predictions = self.model.predict(normalized_data)
        return QualityRating(np.argmax(predictions))  # Index with the highest value is the enum value

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
        self.rating_model = ClassificationModel(self.location_data)
        self.user_data = UserData()

    def user_input(self):
        while True:
            self.user_data.get_user_input()
            rating = self.rating_model.rate(self.user_data)
            print(rating.name)
            exit_loop = input(f"\nDo you want to exit (y/n)")
            if exit_loop.lower() == "y":
                break

    def all_in_sequence(self):
        self.location_data.load()
        self.location_data.preprocess()

        self.rating_model.setup()
        self.rating_model.train(verbose=1)
        self.rating_model.save('model.keras')

        self.rating_model.evaluate()

    def main_menu(self):
        print("Välkommen till områdesklassificering!")

        while True:
            print("\n==============================")
            print("Ange funktion:")
            print("1. Ladda data för träning")
            print("2. Skapa modell") if self.location_data.is_loaded() else None
            print("3. Träna modell") if self.rating_model.model else None
            print("4. Utvärdera modell") if self.rating_model.is_trained and self.location_data.is_loaded() else None
            print("5. Visa graf") if self.rating_model.is_trained else None
            print("6. Spara modell") if self.rating_model.is_trained else None
            print("8. Ladda befintlig modell")
            print("9. Klassificera") if self.rating_model.is_trained else None
            print("0. Avsluta")

            choice = input("Välj ett alternativ: ")
            print("==============================\n")
            if choice == '1':
                self.location_data.load()
                self.location_data.preprocess()
            elif choice == '2':
                self.rating_model.setup()
            elif choice == '3':
                self.rating_model.train(verbose=1)
            elif choice == '4':
                self.rating_model.evaluate()
            elif choice == '5':
                self.rating_model.evaluation_graph()
            elif choice == '6':
                self.rating_model.save('model.keras')
            elif choice == '8':
                self.rating_model.load('model.keras')
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
