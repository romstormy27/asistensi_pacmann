import pandas as pd
from sklearn.model_selection import train_test_split

class dataPreparation:
    
    def __init__(self, path_to_df, df_name='dataset'):
        self.df = pd.read_csv(path_to_df)
        self.df_name = df_name

    # print the dataset
    def print_df(self):
        return(print(self.df))

    # get x and y
    def get_x_y(self, x, y):
        X = self.df[x]
        y = self.df[y]
        return X, y

    # split the dataset
    def split_train_test(self, X, y, with_valid=False, TEST_SIZE=0.2, STRAT=None):

        if with_valid:
            if STRAT:
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE*2, stratify=STRAT, random_state=42)
                x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, stratify=STRAT, random_state=42)
                self.x_train = x_train
                self.x_test = x_test
                self.x_valid = x_valid
                self.y_train = y_train
                self.y_valid = y_valid
                self.y_test = y_test

            else:
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE*2, random_state=42)
                x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
                self.x_train = x_train
                self.x_test = x_test
                self.x_valid = x_valid
                self.y_train = y_train
                self.y_valid = y_valid
                self.y_test = y_test
        else:
            if STRAT:
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=STRAT, random_state=42)
                self.x_train = x_train
                self.x_test = x_test
                self.y_train = y_train
                self.y_test = y_test

            else:
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
                self.x_train = x_train
                self.x_test = x_test
                self.y_train = y_train
                self.y_test = y_test

        # return x_train, x_valid, x_test, y_train, y_valid, y_test

    def print_split_shape(self, with_valid=False):
        if with_valid:
            print(f'x_train: {self.x_train.shape}, y_train: {self.y_train.shape}')
            print(f'x_valid: {self.x_valid.shape}, y_valid: {self.y_valid.shape}')
            print(f'x_test: {self.x_test.shape}, y_test: {self.y_test.shape}')
        else:
            print(f'x_train: {self.x_train.shape}, y_train: {self.y_train.shape}')
            print(f'x_test: {self.x_test.shape}, y_test: {self.y_test.shape}')