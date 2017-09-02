import numbers
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_FILES_PATH = "bank/"


# =======================
#     MENUS FUNCTIONS
# =======================

# Main menu
def show_menu():
    print("Please choose the menu you want to start:")
    print("1. Run Logistic Regression")
    print("2. Exit")
    choice = input(" >>  ")
    exec_menu(choice)
    return


# Execute menu
def exec_menu(choice):
    ch = choice.lower()
    try:
        menu_actions[ch]()
    except KeyError:
        print("Invalid selection, please try again.\n")
        show_menu()
    return


# Exit program
def exit():
    sys.exit()


def load_data_from_csv_file(file_name, sep=';'):
    # return np.genfromtxt(os.path.join(DATA_FILES_PATH,file_name), delimiter=',', dtype=int)
    return pd.read_csv(DATA_FILES_PATH + file_name, sep=sep)


def menu_run_logistic():
    print("menu_run_logistic")
    # get data from CSV
    x_df = load_data_from_csv_file("bank.csv")

    x_df, y_df = make_dummies(x_df)

    length_of_train = round(len(x_df.index) * 0.8)

    x_train, y_train = x_df[:length_of_train], y_df[:length_of_train]
    x_test, y_test = x_df[length_of_train:], y_df[length_of_train:]

    # y_df = np.ravel(load_data_from_csv_file("y_data.csv"))

    # print(list(x_df))
    # print(y_df)
    # print(x_df)
    logistic_object = LogisticRegression(solver='liblinear')
    logistic_object.fit(x_df, y_df)
    score = logistic_object.score(x_train, y_train)
    print("score: " + str(score))

    print(sum(logistic_object.predict(x_test) == y_test), "out of: ", len(y_test.index))
    # logistic_object = LogisticRegression(class_weight={1: 0.9, 0: 0.1})
    # logistic_object = LogisticRegression(solver='lbfgs')
    # logistic_object.fit(x_df, y_df, sample_weight=sample_weight)
    # score = logistic_object.score(x_df, y_df)
    #
    # print("score: " + str(score))

# This function will automatically convert Categories to dummies based on the fact that they contain string on the
# first value of the column. And will convert yes\no fields to 1\0 if the first column will contain values of 'yes'\'no'
# The function will also split y values to separate df
def make_dummies(df):

    dummies_list = []
    dummies_names = []
    # replace yes/no with boolean
    str_to_boolean = {'yes': 1, 'no': 0}

    for column, values in df.iteritems():

        if values[0] in str_to_boolean.keys():
            df[column] = df[column].map(str_to_boolean)

        elif not isinstance(values[0], numbers.Number):
            # make dummies out of textual column and Convert to int32
            dummies_list.append(pd.get_dummies(df[column], prefix=column).astype(dtype='int32'))
            dummies_names.append(column)

    # split y out of the DataFrame
    y_df = df['y']
    # remove y from df
    dummies_names.append('y')

    # # remove column from original df
    df = df.drop(dummies_names, 1)
    # Join both parts
    df = df.join(dummies_list)

    print(df)
    return df, y_df


# =======================
#    MENUS DEFINITIONS
# =======================

# Menu definition
menu_actions = {
    '1': menu_run_logistic,
    '2': exit,
}

# =======================
#      MAIN PROGRAM
# =======================


if __name__ == '__main__':
    # Launch main menu
    # show_menu()
    menu_run_logistic()