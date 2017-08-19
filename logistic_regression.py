import sys, os
import numpy as np
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
    #choice = input(" >>  ")
    #exec_menu(choice)
    exec_menu("1")
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

def load_data_from_csv_file(file_name, sep = ';'):
    # return np.genfromtxt(os.path.join(DATA_FILES_PATH,file_name), delimiter=',', dtype=int)
    return pd.read_csv(DATA_FILES_PATH + file_name, sep=sep)

def menu_run_logistic():
    print("menu_run_logistic")
    # get data from CSV
    x_df = load_data_from_csv_file("bank.csv")

    # replace yes/no with boolean
    str_to_boolean = {'yes': 1, 'no': 0}
    x_df['housing'] = x_df['housing'].map(str_to_boolean)
    x_df['loan'] = x_df['loan'].map(str_to_boolean)
    x_df['default'] = x_df['default'].map(str_to_boolean)
    x_df['y'] = x_df['y'].map(str_to_boolean)

    # split y out of the DataFrame
    y_df = x_df['y']

    print(x_df)
    # make dummies out of textual column
    dummy1 = pd.get_dummies(x_df['marital'], prefix='marital')
    dummy2 = pd.get_dummies(x_df['job'], prefix='job')
    dummy3 = pd.get_dummies(x_df['education'], prefix='education')
    dummy4 = pd.get_dummies(x_df['contact'], prefix='contact')
    dummy5 = pd.get_dummies(x_df['month'], prefix='month')
    dummy6 = pd.get_dummies(x_df['poutcome'], prefix='poutcome')

    # Convert to int
    dummy1 = dummy1.astype(dtype='int32')
    dummy2 = dummy2.astype(dtype='int32')
    dummy3 = dummy3.astype(dtype='int32')
    dummy4 = dummy4.astype(dtype='int32')
    dummy5 = dummy5.astype(dtype='int32')
    dummy6 = dummy6.astype(dtype='int32')

    dummies_list = [dummy1, dummy2, dummy3, dummy4, dummy5, dummy6]

    # remove column from original df
    x_df = x_df.drop(['marital', 'job', 'education', 'contact', 'month', 'poutcome', 'y'], 1)
    # Join both parts
    x_df = x_df.join(dummies_list)

    # y_df = np.ravel(load_data_from_csv_file("y_data.csv"))

    print(list(x_df))
    print(y_df)
    # print(x_df)


    # logistic_object = LogisticRegression(class_weight={1: 0.9, 0: 0.1})
    # logistic_object = LogisticRegression(solver='lbfgs')
    # logistic_object.fit(x_df, y_df, sample_weight=sample_weight)
    # score = logistic_object.score(x_df, y_df)
    #
    # print("score: " + str(score))


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
    show_menu()