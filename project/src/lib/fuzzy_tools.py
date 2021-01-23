# FUZZY CLUSTER MODULE
# v1.0
#
# This module contains functions to support the task of clustering
#   text records into groups of unique itentities. This version is
#   stable and efficient on batches of fewer than 2000 records but
#   the time complexity is polynomial O(x^2) and it is ineffective
#   at clustering records when R > 2000. The following equation is
#   an approximation of the time complexity:
#
#   Time (seconds) = 0.0024(#R)2 + 0.0025(#R) + 2.0395
#
# Future releases will have lower time complexities and will have
#   the ability to cluster more than the 2000 record limit.
#
# AUSTIN H DIAL
# 07/03/2019
#


# ------------------------- IMPORT PACKAGES ------------------------- #

# Standard math libraries
import math as math
import numpy as np
import pandas as pd

# System level control
import sys

# Support for progress bar
import time
from IPython.display import clear_output

# Levenshtein fuzzy comparisons
from fuzzywuzzy import fuzz


# ------------------------- ITERATOR FUNCTION ------------------------- #
# This function operates as an iterator for any data frame. By calling 
#   this function, we pop a record from the dataframe and return it as a
#   series.
#

def iterator(df, destruct, verb):
    """Returns (opt-destructive) a record from a dataframe as a series."""

    # Verbosity and logs
    if verb:
        print('Iterating.')

    # Copy the last record in the deck
    try:
        next_record = df.iloc[len(df) - 1].copy()
    except:
        print('ERROR: ', sys.exc_info()[0], 'in iterate.')

    # Pop the last record from the deck
    if destruct: df.drop(df.index[[next_record.name]], inplace=True)

    # Return next record
    return next_record


# ------------------------- ADJUST WEIGHTS FUNCTION ------------------------- #
# Here we take in the records for comparison and determine what weights must be
#   dropped from the normal weights to insure that the bias is conserved but
#   that the missing values do not eliminate the possibility for a match.
#

def adjust_weights(Norm, RA, RB, verb):
    """Returns adjusted weights for two specific records."""

    # ----- MISSING FEATURES ----- #

    # Check for null in each record

    # Find true indexes
    missing_ind_RA = list(np.where(RA.isnull().values == True)[0])
    if missing_ind_RA:
        if verb:
            print('RA #{} \t: Missing {}'.format(RA.name, missing_ind_RA))

    # Find missing indexes within Group
    missing_ind_RB = list(np.where(RB.isnull().values == True)[0])
    if missing_ind_RB:
        if verb:
            print('Record #{} \t: Missing {}'.format(RB.name, missing_ind_RB))

    # Combine lists of missing indexes
    missing_ind = missing_ind_RA + missing_ind_RB

    # Copy normal weights
    Omega_A = Norm.copy()

    # ----- ADJUSTMENT ----- #

    # Zero out missing indexes
    for i in missing_ind:
        Omega_A[i] = 0

    # Speak if adjusting weights
    if missing_ind:
        if verb:
            print('Adjusting weights.')

    # Divide the weights by the sum of the non-zero weights, print results
    Omega_A = list(Omega_A / round(np.sum([Omega_A[i] for i in np.nonzero(Omega_A)[0]]), 20))

    # Speak if adjusting weights
    if missing_ind:
        if verb:
            print('Omega_A : {}'.format(Omega_A))

    # Check that the weights balance out
    if not math.isclose(np.sum(Omega_A), 1, abs_tol=1e-100):
        if verb:
            print('Weights are not in balance.')
        return -1

    # Return next record
    return Omega_A


# ------------------------- CENTROID ITERATOR FUNCTION ------------------------- #
# When running the comparison system, we are looking to compare new records from
#   the deck against the most representative record from each group. This is
#   conceptually similar to the centroid from the K-Means algorithm. This function
#   iterates through the centroids and returns its corresponding record. This
#   method is not just a function of its own. Rather, it is a demonstration of how
#   the process will operate. The function itself is essentially the iterator
#   function without the destruct option and with the added input of group number:
#   call fifth group leader, then sixth, etc.
#

def centroid_iterator(df, Groups, grp_no, verb):
    """Returns centroid record for a specific group number."""

    # Speak if necessary
    if verb: print('Calling group #{}.'.format(grp_no))

    # Return record for grp_no
    return df.iloc[Groups.loc[Groups[Groups.columns[2]] == True].iloc[grp_no][Groups.columns[1]]]


# ------------------------- SCORE SIMILARITY FUNCTION ------------------------- #
# Now that we have a concise way to readjust the weights with respect to the
#   missing elements in records, we can work on developing a concise function to
#   assess their similarity. This function calls for weight adjustment as well.
#

def score_similarity(df, RA, RB, Omega, verb):
    """Returns the similarity between two records."""

    # Speak if necessary
    if verb: print('Scoring record #{} against #{}.'.format(RA.name, RB.name))

    # Instantiate s
    s = np.array([])

    # Run comparison feature for feature
    for j in range(0, len(df.columns)):
        s = np.append(s, fuzz.ratio(RA.apply(str)[j], RB.apply(str)[j]))

    # Linear product with adjusted weights
    s = np.inner(list(s), adjust_weights(Omega, RA, RB, verb))
    if verb: print('Similarity: {}'.format(s))

    # Return score
    return s


# ------------------------- SCORE AGAINST GROUPS FUNCTION ------------------------- #
# Here we can call the similarity score function in conjuction with the group and
#   dataframe iterators. We will use the iterator non-destructively with the centroid
#   iterator to demonstrate how a record can be popped off of the deck, assess for
#   missing values, scored with Levenshtein Distance against each existing group.
#
# This function returns a list of scores. These scores represent the similarities
#   between the card and each existing group centroid. The index of the max score
#   links to the group number with which the score is associated. This makes it
#   easier to debug the s >= S decisions and ensures that the decisions logic
#   is outside te purview of this function.
#

def score_against_groups(df, Groups, Omega, card, verb):
    """Returns the similarities between a record and each centroid."""

    # Instantiate scores list
    scores = np.array([])

    # Iterate through groups
    for i in range(0, len(Groups.loc[Groups[Groups.columns[2]] == True])):
        # Pop ith group centroid record
        GN = centroid_iterator(df, Groups, i, verb)

        # Score card against ith group
        s = score_similarity(df, card, GN, Omega, verb)

        # Append score to scores list
        scores = np.append(scores, s)

    # Return scores list
    return scores


# ------------------------- CENTROID ASSESSMENT FUNCTION ------------------------- #
# Within the K-means algorithm, the center of a cluster adjusts when new records are
#   assigned. Similarly, we may create individual groups due to uniqueness or
#   incompleteness and then find a more accurate representative for a given
#   identity. Consequently, we must periodically reevaluate which records are
#   centroids.
#
# In the 'Representing Group Centers' notes I describe two measurements for
#   determining the centroid of a group. If the number of records in a group is 2 or
#   fewer then the centroid must be the most complete record: the record with the
#   fewest missing values.* If a group has three or more records then we can take
#   the inner similarity between such records and determine which record is most
#   similar to the records within its set. This way we know that we have selected
#   the best representative of the group.
#
# *Additionally, we could find which record has the longest strings, and is
#   therefore the most detailed but this approach has its drawbacks.
#

def centroid_assessment(df, Groups, Omega, verb):
    """Evaluates existing groups to elect new centroids."""

    # Speak if necessary
    if verb: print('Reassigning centroids.')

    # Iterate through groups
    for i in range(0, np.unique(Groups[Groups.columns[0]].values).size - 1):

        # Select the ith group of records
        Group = Groups.loc[Groups[Groups.columns[0]] == i].copy()

        # Only one record?
        if len(Group) == 1:

            # Assign that one record as the centroid
            Groups.at[Group.loc[Group[Group.columns[2]] == 1].index[0], Group.columns[2]] = 1

        # Two records?
        elif len(Group) == 2:

            # Instantiate missing set
            missing = np.array([])

            # Find missing values for each record
            for r in [0, 1]:
                missing_r = list(np.where(df.iloc[Group.iloc[r][Group.columns[1]]].isnull().values == True)[0])
                missing = np.append(missing, len(missing_r))

            # If there are missing values, which one is missing the fewest features?
            if sum(missing) != 0:
                # Find index of the most complete record
                centroid = [i for i, j in enumerate(missing) if j == min(missing)]

                # Dethrone the current centroid
                Groups.at[Group.loc[Group[Group.columns[2]] == 1].index[0], Groups.columns[2]] = 0

                # Assign the most complete record as the centroid
                Groups.at[Group.iloc[centroid].index[0], Groups.columns[2]] = 1

        # More than 2 records
        elif len(Group) > 2:

            # Instantiate inner similarity set
            inner_similarity = np.zeros((len(Group), len(Group)))

            # Score the similarities of each record
            for rj in range(0, len(Group)):
                for ri in range(0, len(Group)):
                    inner_similarity[rj, ri] = score_similarity(df, df.iloc[Group.iloc[rj][Group.columns[1]]],
                                                                df.iloc[Group.iloc[ri][Group.columns[1]]], Omega, False)

            # Take the average of each column
            candidate_score = np.array([])
            for j in range(0, len(inner_similarity)):
                candidate_score = np.append(candidate_score, np.average(inner_similarity[:, j]))

            # Find the index of the record with the highest commonality
            centroid = [i for i, j in enumerate(candidate_score) if j == max(candidate_score)]

            # Dethrone the current centroid
            Groups.at[Group.loc[Group[Group.columns[2]] == 1].index[0], Groups.columns[2]] = 0

            # Assign the most common record as the centroid
            Groups.at[Group.iloc[centroid].index[0], Groups.columns[2]] = 1

        else:
            print('ERROR: Perverse logic in centroid_assessment.')

    # Return group changes
    return Groups


# ------------------------- APPEND TO GROUPS FUNCTION ------------------------- #
# This function is called to edit the Groups dataframe. It receives a copy of the
#   Groups dataframe and appends a new entry with the group number, record number
#   and a centroid flag then it returns the altered copy. The function can only
#   be called like so
#
#     Groups = append_to_groups( Groups, grp_no, rec_no, new, verb )
#
# Otherwords changes will not be saved to the dataframe as this function only
#   edits and returns a copy. If the record is being added as a new group then
#   the 'new' argument must be set to True so that the new group has a centroid.
#   Otherwise, set the 'new' argument to False to preserve the centroid of an
#   existing group.
#

def append_to_groups(df, grp_no, rec_no, new, verb):
    """Adds a record to the groups dataframe."""

    # Speak if necessary
    if verb:
        if new: print('Appending record #{} as a new group.'.format(rec_no))
        if not new: print('Appending record #{} to group #{}.'.format(rec_no, grp_no))

    # Configure centroid flag
    if new:
        centroid = 1
    else:
        centroid = 0

    # Create temp record for append
    Record = pd.DataFrame(data={df.columns[0]: grp_no, df.columns[1]: rec_no, df.columns[2]: centroid}, index={0})
    df = df.append(Record, ignore_index=True)

    # Return changes
    return df


# ------------------------- PROGRESS BAR FUNCTIONS ------------------------- #
# These functions are designed to support the operation of the progress bar,
#   an ASCII-art graphical visual of the status of the clustering process. The
#   clustering can be arduously slow, necessitating the development of a
#   system to communicate the programs proximity to completion.
#

# Theoretical process with a finite time complexity
def process(ttime):
    # Time delay
    time.sleep(ttime)


# Define progress bar functionality
def update_progress(name, c, number_of_elements):

    # Calculate progress
    progress = c / number_of_elements

    # Define bar
    bar_length = 100
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0

    # Progress fall throughs
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    # Set progress
    block = int(round(bar_length * progress))

    # Update graphics
    clear_output(wait=True)
    text = name + ": [{0}] {1:.5f}% [ {2}/{3} ]".format("#" * block + "-" * (bar_length - block), progress * 100, c, number_of_elements )
    print(text)


# ------------------------- FUZZY CLUSTER SUPER-FUNCTION ------------------------- #
# Using the methods designed in the previous sections, we now compile our work into
#   a single cohesive system of iterating through the dataset, clustering, and
#   returning the final results.
#

def fuzzy_cluster(Master, Deck, Groups, Omega, S, verb):
    # ----- ITERATE THROUGH DECK ----- #

    # Define the number of steps to completion
    number_of_elements = len(Deck)

    # Main loop
    for c in range(0, len(Deck)):

        # Pop card from deck
        card = iterator(Deck, True, False)

        # Speak if necessary
        if verb: print('Card is record #{}'.format(card.name))

        # Obtain scores
        scores = score_against_groups(Master, Groups, Omega, card, verb)

        # Take the highest score information
        max_score_ind = np.argmax(scores)

        # Branch based on s > S?
        if scores[max_score_ind] >= S:

            # Append record to an existing group
            Groups = append_to_groups(Groups, max_score_ind, card.name, False, verb)

        else:

            # Append record as a new group
            Groups = append_to_groups(Groups, np.max(Groups[Groups.columns[0]].values) + 1, card.name, True, verb)

        # Reassess centroids
        #Groups = centroid_assessment(Master, Groups, Omega, verb)

        # Call to update progress bar
        if not verb: update_progress('Clustering', c, number_of_elements)

    # Complete progress bar
    if not verb: update_progress('Clustering', c+1, number_of_elements)

    # Return appended to the Master table
    return Master.join(pd.DataFrame({Groups.columns[0]: list(reversed(Groups[Groups.columns[0]].values.tolist()))}))
