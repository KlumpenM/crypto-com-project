# Cryptographic Computing Project (Privacy Preserving Machine Learning)
This is a project in the course "Cryptographic Computing" at Aarhus University, where there is a mandatory project, and we chose the topic "Privacy Preserving Machine Learning".

## Project description
By running machine learning algortihms inside MPC, you can add privacy in various ways. For instance, when training a model, you can preserve privacy for the training set. Or, when running a classification algorithm, both the input and the model could be kept private. In this project you will study, the techniques for machine learning with MPC, and try to makek a simple, private logistic regression classifier (note: assume pre-trained model, training is out-of-scope). Thjere are 3 key step to look at:
- (1) Implement the arithmetic version with of BeDOZa with multiplcation triples modulo $2^k$.
- (2) secure fixed-point arithmetic for emulating real-number arithmetic with integer operations modulo $2^k$,
- (3) Find a suitable, MPC-friendly approximation to the sigmoid function.

# Requirements
A `requirements.txt` file is available, listing all the necessary python modules to install in order to run the project. <br>
Create a new virtual environment via `python -m venv <enter name of your venv>`. For instance, I named mine as `myenv` <br>
Then activate your virtual environment by entering `myenv/bin/activate` or `source myenv/bin/activate`. <br>
To install the required packages, enter in shell `pip install -r requirements.txt`.