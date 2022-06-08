from script.data_preparation import dataPreparation
from script.preprocessor import Preprocessor

# load dataset
data_object = dataPreparation('dataset/usedcars.csv')

# define the features and the target
x = ["year", "model", "mileage", "color", "transmission"]
y = ["price"]

# get the X and y
X, y = data_object.get_x_y(x=x, y=y)

# split dataframe
data_object.split_train_test(X, y, with_valid=True, TEST_SIZE=0.2)
# print splitted shapes
data_object.print_split_shape(with_valid=True)

numerical_features = ["mileage"]
categorical_features = ["model", "color", "transmission"]

# initiate preprocessor instance
preprocessor = Preprocessor(data_object=data_object)
# get numerics
preprocessor.get_numerical_features(numerical_features)
# get categorics
preprocessor.get_categorical_features(categorical_features)

# fit scaler
preprocessor.fit_scaler()
# transform using fitted scaler
preprocessor.scale()

# fit encoder
preprocessor.fit_encoder()
# encode using fitted encoder
preprocessor.encode()

print(preprocessor.encoded_train)


# # initiate preprocessor instance
# b = Preprocessor(a)
# # get the numerics
# b.get_numerical_features(numerical_features)
# # get the categorics
# b.get_categorical_features(categorical_features)
# # scale the numerics
# ## fit first
# b.fit_scaler()
# ## transform
# b.scale()
# ## print to see
# print(b.train_scaled)