"""
Author: Levi Huillet
Email: huillet@wisc.edu
Course: CS 540
Project: Final Exam: Preliminary Results
"""

import os
from math import log


def train(training_directory, cutoff):
    """loads the training data, estimates the prior distribution P(label)
    and class conditional distributions LaTeX: P(word | label),
    return the trained model"""

    vocab = create_vocabulary(training_directory, cutoff)

    trained_model = {'vocabulary': vocab}

    # load training data
    training_data = load_training_data(vocab, training_directory)

    # create a list of labels
    label_list = os.listdir(training_directory)

    # calculate log priors
    priors = prior(training_data, label_list)

    trained_model['log prior'] = priors

    for label in label_list:
        trained_model['log p(w|y=' + label + ')'] = p_word_given_label(vocab, training_data, label)

    return trained_model


def create_vocabulary(training_directory, cutoff):
    """create and return a vocabulary as a list of word types with counts >= cutoff in the training directory"""
    # holds a list of vocabulary among both training sets
    vocabulary = []
    word_count_bow = []

    # Traverse training_directory / 2016 (not including test)

    training_labels = os.listdir(training_directory)

    # Traverse each training label
    for label in training_labels:
        # get the training set from that label
        training_set = os.listdir(training_directory + "/" + label)

        # for survey response in the training set
        for response in training_set:
            # get the path
            response_file_path = training_directory + "/" + label + "/" + response
            # open the response
            response_file = open(response_file_path, encoding='utf-8')
            # for each line in the response
            for line in response_file:
                if "\n" in line:
                    # remove the trailing '\n'
                    line = line[:-1]
                duplicate_word = False
                # compare it with each line of vocabulary
                for word_index in range(len(vocabulary)):
                    if vocabulary[word_index] == line:
                        # a previously found word type has appeared again; increment count
                        word_count_bow[word_index] += 1
                        duplicate_word = True
                        break
                if not duplicate_word:
                    # add the new word type to the vocabulary and set its count to 1
                    vocabulary.append(line)
                    word_count_bow.append(1)

    # Parse the list according to the cutoff
    vocabulary_length = len(vocabulary)
    word_index = 0
    while word_index < vocabulary_length:
        if word_count_bow[word_index] < cutoff:
            del word_count_bow[word_index]
            del vocabulary[word_index]
            # vocabulary is one element shorter
            vocabulary_length -= 1
            # since vocabulary shortened, word_index + 0 moves to the next element of the list
        else:
            # cut the "\n" off of each string, if applicable
            if "\n" in vocabulary[word_index]:
                vocabulary[word_index] = vocabulary[word_index][:-1]
            # move to the next element of the list
            word_index += 1

    # sort the final vocabulary
    vocabulary.sort()

    return vocabulary


def create_bow(vocab, filepath):
    """create and return a bag of words dictionary from a single survey question"""
    dictionary = {}
    file = open(filepath, encoding='utf-8')

    none_count = 0
    # for each line in the file
    for line in file:
        if "\n" in line:
            # remove the trailing '\n'
            line = line[:-1]
        in_vocab = False
        for word in vocab:
            if line == word:
                in_vocab = True
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1

        if not in_vocab:
            none_count += 1

    if none_count > 0:
        dictionary[None] = none_count

    return dictionary


def load_training_data(vocab, directory):
    """create and return training set (bag-of-words dictionary + label)
    from the survey response files in a training directory"""
    # a list of each BOW and its subdirectory label
    training_data = []

    # Traverse training_directory / 2016 (not including test)

    training_labels = os.listdir(directory)

    # Traverse each training label
    for label in training_labels:
        # get the training set from that label
        training_set = os.listdir(directory + "/" + label)

        # for survey response in the training set
        for response in training_set:
            # get the path
            response_file_path = directory + "/" + label + "/" + response
            # store its BOW representation
            response_bow = create_bow(vocab, response_file_path)
            training_data.append({'label': label, 'bow': response_bow})

    return training_data


def prior(training_data, label_list):
    """given a training set, estimate and return the prior probability p(label) of each label"""

    # initialize the label-count pair list
    label_occurrences = {}
    for label in label_list:
        label_occurrences[label] = 0

    # store each label in the training set with how many times it appeared
    for response in training_data:
        if response['label'] in label_list:
            # increment the count of that label
            label_occurrences[response['label']] += 1
            # {'2016': 5, '2020': 7}

    # calculate the prior probability for each label
    prior_probabilities = {}
    for label in label_occurrences:
        prior_probabilities[label] = log(label_occurrences[label]) - log(len(training_data))

    return prior_probabilities


def p_word_given_label(vocab, training_data, label):
    """given a training set and a vocabulary,
    estimate and return the class conditional distribution LaTeX: P(word|label)
     over all words for the given label using smoothing"""
    # dict word->num_occurrences
    word_occurrences_dict = {}

    for word in vocab:
        word_occurrences_dict[word] = 0

    # tracks the total number of words
    total_words = 0

    # tracks the number of OOV words
    num_oov = 0
    num_oov_type = 0

    # for each set in training_data
    for label_bow in training_data:
        # check if it has the desired the label
        if label_bow['label'] == label:
            # for each word in the bow
            for bow_word in label_bow['bow']:
                word_occurrences = 0
                word_occurrences = label_bow['bow'][bow_word]
                if bow_word in vocab:
                    # add word to dict with word_occurrences
                    word_occurrences_dict[bow_word] += word_occurrences
                else:
                    # None
                    num_oov += word_occurrences
                    num_oov_type = 1
                total_words += word_occurrences

    # add OOV word count to dictionary
    # if num_oov > 0:
    word_occurrences_dict[None] = num_oov

    vocab_length = len(vocab) + 1

    # calculate P(word|label) for each word
    p_word_given_label_dict = {}
    for word in word_occurrences_dict:
        p_word_given_label_dict[word] = log(word_occurrences_dict[word] + 1) - log(total_words + (1 * vocab_length))

    return p_word_given_label_dict


def classify(model, filepath):
    """given a trained model, predict the label for the test document"""
    classification = {}
    label_probs = {}
    new_estimate = 0

    # create a BOW
    bow_of_classify = create_bow(model['vocabulary'], filepath)

    # for each label
    for label in model['log prior']:
        # calculate its argmax
        sum_px_given_y = 0

        # for each word in the bow, add to the sum the word's log from model * the word's count
        for bow_word in bow_of_classify:
            sum_px_given_y += model['log p(w|y=' + label + ')'][bow_word] * bow_of_classify[bow_word]

        # calculate estimate
        new_estimate = sum_px_given_y + model['log prior'][label]

        # add the estimate to the list
        label_probs['log p(y=' + label + '|x)'] = new_estimate

    # find the highest estimate
    max_estimate = 0
    max_label = ''
    for label in model['log prior']:
        if max_estimate == 0:
            max_estimate = label_probs['log p(y=' + label + '|x)']
            max_label = label
        if label_probs['log p(y=' + label + '|x)'] > max_estimate:
            max_estimate = label_probs['log p(y=' + label + '|x)']
            max_label = label

    classification['predicted y'] = max_label

    for label in label_probs:
        classification[label] = label_probs[label]

    return classification


print(classify(model, './prelimFiles/test/at-risk/0.txt'))
print(classify(model, './prelimFiles/test/at-risk/1.txt'))
print(classify(model, './prelimFiles/test/at-risk/2.txt'))
print(classify(model, './prelimFiles/test/at-risk/3.txt'))
print(classify(model, './prelimFiles/test/not-at-risk/0.txt'))
print(classify(model, './prelimFiles/test/not-at-risk/1.txt'))
print(classify(model, './prelimFiles/test/not-at-risk/2.txt'))

