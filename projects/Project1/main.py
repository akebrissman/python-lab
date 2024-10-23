import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model


class LocationData:
    def __init__(self):
        self.locations_df = None
        self.features_df = None
        self.target_df = None
        self.data = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.number_of_rows = 0
        self.number_of_features = 0
        self.number_of_output_categories = 0
        self.feature_matrix = None
        self.preprocessor = None
        self.normalization_constant = 1

    def is_loaded(self):
        return True if self.number_of_rows > 0 else False

    def load(self):
        try:
            # Load data from csv file
            file_name = "location_training_data.csv"
            current_directory = os.path.dirname(__file__)
            file_path = os.path.join(current_directory, file_name)
            self.locations_df = pd.read_csv(file_path, delimiter=';', decimal=',')

            # Drop blank lines
            self.locations_df.dropna(how="all", inplace=True)

            self.features_df = self.locations_df[["payed_parking", "apartments", "apartment_age"]]
            self.target_df = self.locations_df["rank"]

            print(f"{len(self.locations_df)} lines of data were loaded successfully. ")
        except Exception as e:
            print(e)

    def preprocess(self):
        # Preprocessing dataframe:
        # self.number_of_rows = len(self.features_df)
        # self.number_of_features = len(self.features_df.columns)

        self.number_of_rows = self.features_df.shape[0]
        self.number_of_features = self.features_df.shape[1]
        self.number_of_output_categories = 3  # BRA(0), MEDEL(1), DÅLIG(2) TODO: skapa enum

        self.data = self.features_df.to_numpy()
        self.target = self.target_df.to_numpy()

        X, y = self.data, self.target

        # Dela upp datan i 20% testdata, 80% träningsdata
        train_test_split_data = train_test_split(X, y, test_size=0.2, random_state=42)

        # X är datat, y är labels (korrekt output/prediction för det datan)
        X_train = train_test_split_data[0]  # Träningsdata
        X_test = train_test_split_data[1]  # Testdata
        y_train = train_test_split_data[2]  # Tränings-labels
        y_test = train_test_split_data[3]  # Test-labels
        # Samma som:
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalisera indata
        # Viktigt för att få alla features i samma skala
        # TODO: Use lines below when the model works fine. ChatGPT tycker att man skall dela både train och test med X_train
        self.normalization_constant = np.max(X_train)
        self.X_train = X_train  # / np.max(X_train)
        self.X_test = X_test  # / np.max(X_test)

        # Konvertera etiketter till one-hot encoding
        # Nödvändigt för fler flerklass-klassificering
        self.y_train = tf.keras.utils.to_categorical(y_train)
        self.y_test = tf.keras.utils.to_categorical(y_test)

        print("Data preprocessed successfully.")
        print(f"Längd på träningsdata {len(X_train)}, maxvalue {np.max(X_train)} ")
        print(f"Längd på testdata {len(X_test)}, maxvalue {np.max(X_test)} ")


class UserData:
    def __init__(self):
        self.payed_parking = 0
        self.apartments = 0
        self.apartment_age = 0

    def get_user_input(self):

        while True:
            try:
                value = float(input(f"Please enter payed parking value (0-10): "))
                if 0 <= value <= 100:
                    self.payed_parking = value
                    break
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")

        while True:
            try:
                value = float(input(f"Please enter number of apartments (1-5): "))
                if 0 <= value <= 100:
                    self.apartments = value
                    break
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")

        while True:
            try:
                value = float(input(f"Please enter apartment age value (1-5): "))
                if 0 <= value <= 100:
                    self.apartment_age = value
                    break
                else:
                    print("Please enter a number between 1 and 5.")

            except ValueError:
                print("Please enter a valid number.")


class ClassificationModel:
    def __init__(self, location_data=None):
        self.location_data = location_data
        self.model: tf.keras.models.Sequential = None
        self.history = None
        self.normalization_constant = location_data.normalization_constant if location_data else 1
        self.is_trained = False

    def setup(self):
        # Skapa en enkel sekventiell modell.
        # Linjär stack av lager, passar detta relativt simpla klassificierings-problemet.
        # Grundläggande typ av neural network, varje lager kopplat till nästa
        # Input-Layer: Specificera input-shape = antal features för varje blomma, d.v.s datats dimension = 4
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
            accuracy: 0.3421: Detta är träningsnoggrannheten för denna epoch. Här är det 34.21%, vilket innebär att modellen korrekt klassificerade 34.21% av träningsdata.
            loss: 2.8861: Detta är förlustvärdet (loss) för modellen under träningen. Lägre värden indikerar bättre prestanda. Det här värdet är relativt högt, vilket tyder på att modellen har svårt att göra korrekta förutsägelser.
            val_accuracy: 0.3000: Detta är noggrannheten på valideringsdatasetet, vilket är 30%. Detta värde används för att utvärdera hur bra modellen generaliserar på osedd data.
            val_loss: 2.7057: Detta är förlustvärdet på valideringsdatasetet. Även här, ju lägre, desto bättre.
            Riktvärden för att tolka resultaten:
            Accuracy: Generellt sett, om noggrannheten över 70% kan anses vara acceptabelt, beroende på problemet. För mer komplexa problem kan man behöva över 80%. I ditt fall är 34.21% och 30% relativt låga.
            Loss: Ett förlustvärde under 1.0 är ofta en bra indikator, men det beror på vilken typ av problem du arbetar med. Ju lägre, desto bättre."""
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

        # TODO: Kanske är bättre att använda 50% av träningsdatat än testdatat som representerar 20% av all data.
        target_accuracy = 0.80
        num_epochs = 50
        while True:
            result = self.model.fit(self.location_data.X_train,
                                          self.location_data.y_train,
                                          validation_data=(self.location_data.X_test, self.location_data.y_test),
                                          epochs=num_epochs,
                                          callbacks=[early_stopping],
                                          verbose=verbose)

            accuracy = result.history["accuracy"][-1]
            val_accuracy = result.history['val_accuracy'][-1]
            self.history = result.history

            print("")
            print(f"Acceptabel valideringsnoggrannhet {target_accuracy:.2f} \n"
                  f"Valideringsnoggrannhet {val_accuracy:.2f} \n"
                  f"Noggrannhet {accuracy:.2f}")

            if val_accuracy >= target_accuracy:
                self.is_trained = True
                break
            else:
                # Gör justeringar här, t ex, ändra hyperparametrar eller öka data
                # target_accuracy -= 0.10
                print("För dåligt, kör igen")

        print(f"Model trained successfully.")

    def evaluation_graph(self):
        # Visualisera träningshistorik
        plt.figure(figsize=(12, 4))

        # Plot träningsnoggrannhet och valideringsnoggrannhet
        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot träningsförlust och valideringsförlust
        # Training Loss: Beräknas under varje epoch. Representerar hur bra model fitting.
        # Validation Loss: Beräknas på separat validerings-data. Visar eventuell model overfitting.
        # Om båda går nedåt under träning, lär sig modellen väl.
        # Om training loss går ner, men validation loss börjar gå upp, kan det indikera overfitting
        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def evaluate(self):
        # Utvärdera modellen på testdata
        test_loss, test_acc = self.model.evaluate(self.location_data.X_test, self.location_data.y_test, verbose=2)
        print(f'Test Accuracy: {test_acc}, Test Loss {test_loss}')

        # Gör prediktioner på några exempel
        sample_predictions = self.model.predict(self.location_data.X_test[:3], verbose=1)
        print("\nSample predictions:")
        for i, prediction in enumerate(sample_predictions):
            print(f"Example {i + 1}: {prediction}")
            print(f"Predicted class: {np.argmax(prediction)}")
            print(f"Actual class: {np.argmax(self.location_data.y_test[i])}")
            print(f"Result: {np.argmax(self.location_data.y_test[i]) == np.argmax(prediction)}")
            print()

    def get_rank(self, user_preferences: UserData):
        values = np.array(
            [[user_preferences.payed_parking, user_preferences.apartments, user_preferences.apartment_age]])
        normalized_data = values / self.normalization_constant
        prediction = self.model.predict(normalized_data)
        return np.argmax(prediction)  # Index with the highest value

    def save(self, file_name='model.h5'):
        file_type = file_name.split('.')[1]
        if file_type == "keras":
            # Spara modellen i keras-format
            self.model.save(file_name)
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

    def load(self, file_name="model.h5"):
        file_type = file_name.split('.')[1]
        if file_type == "keras":
            # Ladda modellen
            self.model = load_model(file_name)
            self.is_trained = True
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


class UserInterface:
    def __init__(self, recommendation_model, user_data):
        self.recommendation_model = recommendation_model
        self.user_data = user_data

    def run(self):
        print("Welcome to the Location Rank predictor!")

        while True:
            self.user_data.get_user_input()
            rank = self.recommendation_model.get_rank(self.user_data)
            print(rank)
            exit_loop = input(f"\nDo you want to exit (y/n)")
            if exit_loop.lower() == "y":
                break


def main():
    # Load and prepare Location data
    location_data = LocationData()
    location_data.load()
    location_data.preprocess()

    # Initialize and train the model
    rank_model = ClassificationModel(location_data)
    rank_model.setup()
    rank_model.train(verbose=1)
    rank_model.save('rank_model.keras')

    rank_model.evaluate()
    # rank_model.evaluation_graph()

    # rank_model.get_rank(1,1,1)

    # Ask the user for data
    user_data = UserData()

    # Run the user interface
    ui = UserInterface(rank_model, user_data)
    ui.run()


def use_existing_model():
    rank_model = ClassificationModel()
    rank_model.load('rank_model.keras')
    user_data = UserData()

    # Run the user interface
    ui = UserInterface(rank_model, user_data)
    ui.run()


def menu_load_data(location_data):
    location_data.load()
    location_data.preprocess()


def menu_setup_model(rank_model):
    # Initialize and train the model
    rank_model.setup()


def menu_train_model(rank_model):
    rank_model.train(verbose=1)


def menu_save_model(rank_model):
    rank_model.save('rank_model.keras')


def menu_load_model(rank_model):
    rank_model.load('rank_model.keras')


def main_menu():
    location_data = LocationData()
    rank_model = ClassificationModel(location_data)
    user_data = UserData()

    while True:
        print("\n============================")
        print("Ange funktion:")
        print("1. Ladda data för träning")
        print("2. Skapa modell") if location_data.is_loaded() else None
        print("3. Träna modell") if rank_model.model else None
        print("4. Utvärdera modell") if rank_model.is_trained and location_data.is_loaded() else None
        print("5. Visa graf") if rank_model.is_trained else None
        print("6. Spara modell") if rank_model.is_trained else None
        print("8. Ladda befintlig modell")
        print("9. Testa") if rank_model.is_trained else None
        print("0. Avsluta")

        choice = input("Välj ett alternativ: ")
        print("============================")
        if choice == '1':
            menu_load_data(location_data)
        elif choice == '2':
            menu_setup_model(rank_model)
        elif choice == '3':
            menu_train_model(rank_model)
        elif choice == '4':
            rank_model.evaluate()
        elif choice == '5':
            rank_model.evaluation_graph()
        elif choice == '6':
            menu_save_model(rank_model)
        elif choice == '8':
            menu_load_model(rank_model)
        elif choice == '9':
            # Run the user interface
            ui = UserInterface(rank_model, user_data)
            ui.run()
        elif choice == '0':
            print("Avslutar programmet.")
            break
        else:
            print("Ogiltigt alternativ, försök igen.")


if __name__ == "__main__":
    main_menu()