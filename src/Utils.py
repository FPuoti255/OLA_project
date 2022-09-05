import warnings
warnings.filterwarnings("ignore")


debug = False
def log(msg):
    if(debug):
        print(msg)

# -------------------------------
# ---Project General Constants---
# -------------------------------
NUM_OF_PRODUCTS = 5
NUM_OF_USERS_CLASSES = 3

# probability of observing the second slot of suggested products. Assumed to be known in the project.
# (al momento ho messo 0.8 come numero perhé è il primo che mi è venuto in mente ma lo possiamo cambiare)
LAMBDA = 0.6