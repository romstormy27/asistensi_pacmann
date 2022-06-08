from script.data_preparation import dataPreparation
from script.preprocessor import Preprocessor

# load dataset
a = dataPreparation('dataset/usedcars.csv')

# define the features and the target
x = ["year", "model", "mileage", "color", "transmission"]
y = ["price"]

# get the X and y
X, y = a.get_x_y(x=x, y=y)

# split dataframe
a.split_train_test(X, y, with_valid=True, TEST_SIZE=0.2)
# print splitted shapes
# a.print_split_shape(with_valid=True)

numerical_features = ["mileage"]
categorical_features = ["model", "color", "transmission"]

# initiate preprocessor instance
b = Preprocessor(a)
# get the numerics
b.get_numerical_features(numerical_features)
# get the categorics
b.get_categorical_features(categorical_features)
# scale the numerics
## fit first
b.fit_scaler()
## transform
b.scale()
## print to see
print(b.train_scaled)