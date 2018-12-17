# CS466 Final Project - CT HMM for ALS Progression

Michael Colomb, Brian Luc, Ophir Sneh, and Sam Cheung's final project for CS466.

We used a Continuous-Time Hidden Markov Model to model the progression of
Amytrophic Lateral Sclerosis (ALS). More information about the project and our
findings can be found in our paper.

## Usage

The following command can be used to test/train our model. More information can be found by using the `-h` flag.
```
usage: main.py [-h] [-M N_STATES] [-e EPOCHS] [-s SAVE_FILENAME]
               [-l LOAD_FILENAME] [-t SAVE_EPOCHS] [-p PLOT_FILENAME]
               [-n NUM_PIDS] [-r]
               data_csv
```

NOTE: Make sure you are using python 3.7.
