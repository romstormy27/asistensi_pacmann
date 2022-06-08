from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

class Preprocessor:

    def __init__(self, data_object):
        self.x_train = data_object.x_train
        self.x_valid = data_object.x_valid
        self.x_test = data_object.x_test

    def get_numerical_features(self, X):
        numerical_features = X
        self.train_numerics = self.x_train[numerical_features]
        self.valid_numerics = self.x_valid[numerical_features]
        self.test_numerics = self.x_test[numerical_features]

    def get_categorical_features(self, X):
        categorical_features = X
        self.train_categorics = self.x_train[categorical_features]
        self.valid_categorics = self.x_valid[categorical_features]
        self.test_categorics = self.x_test[categorical_features]

    def fit_scaler(self):
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.train_numerics)

    def scale(self):
        self.train_scaled = self.scaler.transform(self.train_numerics)
        self.valid_scaled = self.scaler.transform(self.valid_numerics)
        self.test_scaled = self.scaler.transform(self.test_numerics)

    def fit_standardizer(self, train_numerics):
        pass

    def standardize(self, X):
        pass
    
    def fit_encoder(self, train_categorical):
        pass

    def encode(self, X):
        pass