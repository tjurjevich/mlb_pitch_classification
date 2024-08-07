import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from tensorflow import keras, math
from tensorflow.keras import layers, callbacks
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

warnings.filterwarnings(action='ignore', category=UserWarning)


# Load statcast training data.
def upload_data():
    file_loc = f'{os.getcwd()}/statcast_data_TRAIN.csv'
    return pd.read_csv(file_loc)

# Reclassify pitches that are seen very rarely. Define globally to use for future predictions if needed.
def reclassify_pitches(data: pd.core.frame.DataFrame):
    """
    Two pitch types occur <10 times over 10,000+ documented pitches. For sake of this model,
    these are getting converted to a similar pitch.
    """
    copy = data.copy()
    return copy.replace({'pitch_type':{'CS':'CU', 'PO':'FF'}})

class NNPitchClassification:
    """
    PitchClassificationNN will model and store neural network information for pitch classification.

    7 input variables are used to train this model:
        1) release_speed: the velocity of the pitch
        2) pfx_x: horizontal pitch movement
        3) pfx_z: vertical pitch movement
        4) spin_axis: the relative direction of the pitch's rotation
        5) release_spin_rate: the spin rate of the pitch on it's spin_axis
        6) p_throws: either R (right) or L (left)
        7) player_name: last name, first name of the pitcher

    1 output variable comprised of n logits (# pitch types)

    """
    # Define model input/output
    all_predictors = ['release_speed', 'pfx_x', 'pfx_z', 'spin_axis', 'release_spin_rate', 'p_throws', 'player_name']
    num_predictors = ['release_speed', 'pfx_x', 'pfx_z', 'spin_axis', 'release_spin_rate']
    cat_predictors = ['p_throws', 'player_name']
    output_variable = 'pitch_type'

    def __init__(self, data: pd.core.frame.DataFrame):
        self.raw_data = reclassify_pitches(data)

    def process_input(self):
        """
        Standardize numerical variables and one-hot encode categorical variables.
        """
        print('\tEncoding and standardizing input variables...')
        self.raw_input = self.raw_data[['release_speed', 'pfx_x', 'pfx_z', 'spin_axis', 'release_spin_rate', 'p_throws', 'player_name']]
        num_transform = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
        cat_transform = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
        self.InputPreprocessor = make_column_transformer(
            (num_transform, self.num_predictors),
            (cat_transform, self.cat_predictors)
        )

    def encode_output(self):
        """
        Label encode pitch_type.
        """
        print('\tEncoding pitch type...')
        self.raw_output = self.raw_data['pitch_type']
        self.PitchEncoder = LabelEncoder()
        self.encoded_output = self.PitchEncoder.fit_transform(self.raw_output)

    def resample(self):
        """
        Use combination of over- and under- sample to evenly distribute pitch types for NN training. Eliminated
        for the time being due to excessive completion time.
        """
        print('\tApplying resample methods...')
        smote = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def prepare_training_data(self, train_size: float, reproducible: bool):   
        print('\nBeginning data clean process.\n')  
        # Standardize/encode/resample
        self.process_input()
        self.encode_output()
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.raw_input, self.encoded_output, stratify=self.encoded_output, train_size=train_size, random_state=reproducible)
        self.X_train = self.InputPreprocessor.fit_transform(self.X_train)
        self.X_valid = self.InputPreprocessor.transform(self.X_valid) 
        # self.resample()
        # Store input/output shapes for model fit/compilation
        self.output_shape = len(list(set(self.encoded_output)))
        self.input_shape = self.X_train.shape[1]
        print(f'\nData cleaning complete.\n\tTraining set size: {len(self.y_train)}.\n\tValidation set size: {len(self.y_valid)}.\n')

    def build_model(self, num_layers: int, num_neurons: int, add_normalization: bool):
        """
        Create model based on supplied parameters.
        """
        if num_layers < 2:
            raise Exception("2 or more layers are necessary to create neural network.")
        # Define base model to build upon.
        self.model = keras.Sequential()
        # Add input layer
        self.model.add(layers.Dense(num_neurons, activation='relu', kernel_initializer='random_normal', input_dim = self.input_shape))
        # Minimum case of 2 layers (input/output layer)
        if num_layers == 2:
            self.model.add(layers.Dense(self.output_shape, activation='softmax'))
        else:
            for i in range(num_layers-2):
                if add_normalization:
                    self.model.add(layers.BatchNormalization())
                    self.model.add(layers.Dense(num_neurons, activation='relu'))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Dense(self.output_shape, activation='softmax'))

    def compile_model(self):
        self.model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )
        print('\nModel successfully compiled.\n')
    
    def train_model(self, batch_size: int, epochs: int, add_early_stopping: bool):
        print('\nBeginning model training. This may take a few minutes...\n')
        if add_early_stopping:
            early_stopping = [callbacks.EarlyStopping(
            min_delta = 0.001,
            patience = 10,
            restore_best_weights = True
            )]
        else:
            early_stopping = None
        self.epoch_history = self.model.fit(
        self.X_train, self.y_train,
        validation_data=(self.X_valid, self.y_valid),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=early_stopping
        )
        self.probability_model = keras.Sequential([self.model, keras.layers.Softmax()])

        print('\nModel training completed.\n')
    
    def plot_training_metrics(self):
        """
        Plot each epoch and its training/validation accuracy and loss.
        """
        history = pd.DataFrame(self.epoch_history.history)
        fig, axs = plt.subplots(2,1)
        axs[0].plot(history[['loss','val_loss']], label=['loss','val_loss'])
        axs[0].set_title('Categorical Cross-entropy')
        axs[1].plot(history[['accuracy','val_accuracy']], label=['accuracy','val_accuracy'])
        axs[1].set_title('Accuracy')
        plt.show()
    
    def make_new_predictions(self, input: pd.core.frame.DataFrame, inv_transform: bool):
        enc_input = self.InputPreprocessor.transform(input[['release_speed', 'pfx_x', 'pfx_z', 'spin_axis', 'release_spin_rate', 'p_throws', 'player_name']])
        if inv_transform:
            return self.PitchEncoder.inverse_transform(np.argmax(self.probability_model.predict(enc_input), axis=1))
        else:
            return np.argmax(self.probability_model.predict(enc_input), axis=1)
    
    def get_new_data_accuracy(self, testdata: pd.core.frame.DataFrame, include_confusion_matrix: bool):
        enc_input = self.InputPreprocessor.transform(testdata[['release_speed', 'pfx_x', 'pfx_z', 'spin_axis', 'release_spin_rate', 'p_throws', 'player_name']])
        y_true = self.PitchEncoder.transform(testdata['pitch_type'])
        probability_model = keras.Sequential([self.model, keras.layers.Softmax()])
        y_pred = np.argmax(probability_model.predict(enc_input), axis=1)
        if include_confusion_matrix:
            print(f'\nTest data accuracy: {accuracy_score(y_true=y_true, y_pred=y_pred)}.\n')
            confusion = math.confusion_matrix(labels=y_true, predictions=y_pred)
            print(f'\nConfusion matrix (array): {confusion}')
            ax = seaborn.heatmap(confusion, square=True, annot=True, annot_kws={'fontsize':6}, fmt='d', xticklabels=self.PitchEncoder.classes_, yticklabels=self.PitchEncoder.classes_)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            plt.show()
        else:
            print(f'\nNew data accuracy: {accuracy_score(y_true=y_true, y_pred=y_pred)}.\n')
            


if __name__ == '__main__':
    """
    Create model instance with following parameters: split dataset into 70% training/30% testing, add 4 dense layers to NN (each containing 256 units),
    and fit model with batch sizes of 512 samples using 100 epochs (while using early stopping measures).
    """

    data = upload_data()
    myModel = NNPitchClassification(data)
    myModel.prepare_training_data(0.7, True)
    myModel.build_model(4, 256, True)
    myModel.compile_model()
    myModel.train_model(batch_size=512, epochs=100, add_early_stopping=True)

    """
    # Test on new data
    new_data = pd.read_csv(f'{os.getcwd()}/statcast_data_TEST.csv')
    new_data = reclassify_pitches(new_data)
    myModel.get_new_data_accuracy(new_data, include_confusion_matrix=True)
    """

    # Print model summary
    print(myModel.model.summary())

    """
    # Save model for future use, if desired
    myModel.model.save('pitch_classifier_neural_network.keras')
    """


