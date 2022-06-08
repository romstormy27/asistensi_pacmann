from script.data_preparation import dataPreparation

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
a.print_split_shape(with_valid=True)