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

    dummy1 = pd.get_dummies(x_df['marital'])

    # x_df = x_df[list(x_df).remove('marital')].join(dummy1)
    # print(dummy1)
    # y_df = np.ravel(load_data_from_csv_file("y_data.csv"))

    print(x_df)

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