import collections
import re
import sys
import unicodedata

import nltk


def readDict():
    dictionaryPath = "LIWC2015_English.dic"
    catList = collections.OrderedDict()
    catLocation = []
    wordList = {}
    finalDict = collections.OrderedDict()

    # Check to make sure the dictionary is properly formatted
    with open(dictionaryPath, "r") as dictionaryFile:
        for idx, item in enumerate(dictionaryFile):
            if "%" in item:
                catLocation.append(idx)
        if len(catLocation) > 2:
            # There are apparently more than two category sections; throw error and die
            sys.exit("Invalid dictionary format. Check the number/locations of the category delimiters (%).")

    # Read dictionary as lines
    with open(dictionaryPath, "r") as dictionaryFile:
        lines = dictionaryFile.readlines()

    # Within the category section of the dictionary file, grab the numbers associated with each category
    for line in lines[catLocation[0] + 1:catLocation[1]]:
        try:
            if re.split(r'\t+', line)[0] == '':
                catList[re.split(r'\t+', line)[1]] = [re.split(r'\t+', line.rstrip())[2]]
            else:
                catList[re.split(r'\t+', line)[0]] = [re.split(r'\t+', line.rstrip())[1]]
        except:  # likely category tags
            pass

    # Now move on to the words
    for idx, line in enumerate(lines[catLocation[1] + 1:]):
        # Get each line (row), and split it by tabs (\t)
        workingRow = re.split('\t', line.rstrip())
        wordList[workingRow[0]] = list(workingRow[1:])

    # Merge the category list and the word list
    for key, values in wordList.items():
        if "(" in key and ")" in key:
            key = key.replace("(", "").replace(")", "")
        # these words are ambiguous and cause errors
        if key == "kind" or key == "like":
            continue
        if not key in finalDict:
            finalDict[key] = []
        for catnum in values:
            try:  # catch errors (e.g. with dic formatting)
                workingValue = catList[catnum][0]
                finalDict[key].append(workingValue)
            except:
                print(catnum)
    return (finalDict, catList.values())


def wordCount(data, dictOutput, catList):
    # Create a new dictionary for the output
    outList = collections.OrderedDict()

    # Number of non-dictionary words
    nonDict = 0

    # Convert to lowercase
    data = data.lower()

    # Tokenize and create a frequency distribution

    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(data)

    fdist = nltk.FreqDist(tokens)
    wc = len(tokens)

    # bad stems
    bad_stems = []

    # Using the Porter stemmer for wildcards, create a stemmed version of the data
    porter = nltk.PorterStemmer()
    stems = [porter.stem(word) for word in tokens]
    # handle bad stems
    # some words get counted twice due to created stem being different from the actual word
    # e.g. "happy" gets stemmed to "happi*" which produces an additional distinct match later on
    # so we fix this
    for stem in stems:
        good_token = False
        for token in tokens:
            if stem in token:
                good_token = True
        if good_token == False:
            bad_stems.append(stem)

    fdist_stem = nltk.FreqDist(stems)

    # Access categories and populate the output dictionary with keys
    for cat in catList:
        outList[cat[0]] = 0

    # Dictionaries are more useful
    fdist_dict = dict(fdist)
    fdist_stem_dict = dict(fdist_stem)
    # print(bad_stems)
    for stem in bad_stems:
        fdist_stem_dict.pop(stem, None)
    # print(fdist_stem_dict)

    # Number of classified words
    classified = 0

    for key in dictOutput:
        if "*" in key and key[:-1] in fdist_stem_dict:
            classified = classified + fdist_stem_dict[key[:-1]]
            for cat in dictOutput[key]:
                outList[cat] = outList[cat] + fdist_stem_dict[key[:-1]]
        elif key in fdist_dict:
            classified = classified + fdist_dict[key]
            for cat in dictOutput[key]:
                outList[cat] = outList[cat] + fdist_dict[key]

    # Calculate the percentage of words classified
    if wc > 0:
        percClassified = (float(classified) / float(wc)) * 100
    else:
        percClassified = 0

    # Return the categories, the words used, the word count, the number of words classified, and the percentage of words classified.
    return [outList, tokens, wc, classified, percClassified]


dictIn, catList = readDict()
# run the wordCount function

def liwc(text):
    return wordCount(text, dictIn, catList)

contraction_mapping = {"’": "'", "RT ": " ", "ain't": "is not", "aren't": "are not", "can't": "can not",
                       "'cause": "because", "could've": "could have",
                       "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
                       "hadn't": "had not",
                       "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
                       "he's": "he is",
                       "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                       "I'd": "I would",
                       "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                       "I've": "I have",
                       "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
                       "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                       "it'll": "it will",
                       "it'll've": "it will have", "it's": "it is", "it’s": "it is", "let's": "let us",
                       "ma'am": "madam", "mayn't": "may not",
                       "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
                       "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                       "needn't've": "need not have",
                       "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                       "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                       "she'd've": "she would have",
                       "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                       "should've": "should have",
                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                       "so's": "so as",
                       "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                       "here's": "here is",
                       "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                       "they'll've": "they will have",
                       "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                       "we'd": "we would",
                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                       "we've": "we have",
                       "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                       "what're": "what are",
                       "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                       "where'd": "where did",
                       "where's": "where is", "where've": "where have", "who'll": "who will",
                       "who'll've": "who will have",
                       "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                       "will've": "will have",
                       "won't": "will not", "won't've": "will not have", "would've": "would have",
                       "wouldn't": "would not",
                       "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                       "y'all'd've": "you all would have",
                       "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                       "you'd've": "you would have",
                       "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have",
                       "It's": "It is", "You'd": "You would",
                       ' u ': " you ", 'yrs': 'years', 'FYI': 'For your information', ' im ': ' I am ', 'lol': 'LOL',
                       'You\'re': 'You are'
    , 'can’t': 'can not', '…': '. ', '...': '. ', '\'\'': '\'', '≠': '', 'ain’t': 'am not', 'I’m': 'I am', 'RT\'s': ''}
emoticons = {
    ':*': '<kiss>',
    ':-*': '<kiss>',
    ':x': '<kiss>',
    ':-)': '<happy>',
    ':-))': '<happy>',
    ':-)))': '<happy>',
    ':-))))': '<happy>',
    ':-)))))': '<happy>',
    ':-))))))': '<happy>',
    ':)': '<happy>',
    ':))': '<happy>',
    ':)))': '<happy>',
    ':))))': '<happy>',
    ':)))))': '<happy>',
    ':))))))': '<happy>',
    ':)))))))': '<happy>',
    ':o)': '<happy>',
    ':]': '<happy>',
    ':3': '<happy>',
    ':c)': '<happy>',
    ':>': '<happy>',
    '=]': '<happy>',
    '8)': '<happy>',
    '=)': '<happy>',
    ':}': '<happy>',
    ':^)': '<happy>',
    '|;-)': '<happy>',
    ":'-)": '<happy>',
    ":')": '<happy>',
    '\o/': '<happy>',
    '*\\0/*': '<happy>',
    ':-D': '<laugh>',
    ':D': '<laugh>',
    '8-D': '<laugh>',
    '8D': '<laugh>',
    'x-D': '<laugh>',
    'xD': '<laugh>',
    'X-D': '<laugh>',
    'XD': '<laugh>',
    '=-D': '<laugh>',
    '=D': '<laugh>',
    '=-3': '<laugh>',
    '=3': '<laugh>',
    'B^D': '<laugh>',
    '>:[': '<sad>',
    ':-(': '<sad>',
    ':-((': '<sad>',
    ':-(((': '<sad>',
    ':-((((': '<sad>',
    ':-(((((': '<sad>',
    ':-((((((': '<sad>',
    ':-(((((((': '<sad>',
    ':(': '<sad>',
    ':((': '<sad>',
    ':(((': '<sad>',
    ':((((': '<sad>',
    ':(((((': '<sad>',
    ':((((((': '<sad>',
    ':(((((((': '<sad>',
    ':((((((((': '<sad>',
    ':-c': '<sad>',
    ':c': '<sad>',
    ':-<': '<sad>',
    ':<': '<sad>',
    ':-[': '<sad>',
    ':[': '<sad>',
    ':{': '<sad>',
    ':-||': '<sad>',
    ':@': '<sad>',
    ":'-(": '<sad>',
    ":'(": '<sad>',
    'D:<': '<sad>',
    'D:': '<sad>',
    'D8': '<sad>',
    'D;': '<sad>',
    'D=': '<sad>',
    'DX': '<sad>',
    'v.v': '<sad>',
    "D-':": '<sad>',
    '(>_<)': '<sad>',
    ':|': '<sad>',
    '>:O': '<surprise>',
    ':-O': '<surprise>',
    ':-o': '<surprise>',
    ':O': '<surprise>',
    '°o°': '<surprise>',
    'o_O': '<surprise>',
    'o_0': '<surprise>',
    'o.O': '<surprise>',
    'o-o': '<surprise>',
    '8-0': '<surprise>',
    '|-O': '<surprise>',
    ';-)': '<wink>',
    ';)': '<wink>',
    '*-)': '<wink>',
    '*)': '<wink>',
    ';-]': '<wink>',
    ';]': '<wink>',
    ';D': '<wink>',
    ';^)': '<wink>',
    ':-,': '<wink>',
    '>:P': '<tong>',
    ':-P': '<tong>',
    ':P': '<tong>',
    'X-P': '<tong>',
    'x-p': '<tong>',
    ':-p': '<tong>',
    ':p': '<tong>',
    '=p': '<tong>',
    ':-Þ': '<tong>',
    ':Þ': '<tong>',
    ':-b': '<tong>',
    ':b': '<tong>',
    ':-&': '<tong>',
    '>:\\': '<annoyed>',
    '>:/': '<annoyed>',
    ':-/': '<annoyed>',
    ':-.': '<annoyed>',
    ':/': '<annoyed>',
    ':\\': '<annoyed>',
    '=/': '<annoyed>',
    '=\\': '<annoyed>',
    ':L': '<annoyed>',
    '=L': '<annoyed>',
    ':S': '<annoyed>',
    '>.<': '<annoyed>',
    ':-|': '<annoyed>',
    '<:-|': '<annoyed>',
    ':-X': '<seallips>',
    ':X': '<seallips>',
    ':-#': '<seallips>',
    ':#': '<seallips>',
    'O:-)': '<angel>',
    '0:-3': '<angel>',
    '0:3': '<angel>',
    '0:-)': '<angel>',
    '0:)': '<angel>',
    '0;^)': '<angel>',
    '>:)': '<devil>',
    '>:D': '<devil>',
    '>:-D': '<devil>',
    '>;)': '<devil>',
    '>:-)': '<devil>',
    '}:-)': '<devil>',
    '}:)': '<devil>',
    '3:-)': '<devil>',
    '3:)': '<devil>',
    'o/\o': '<highfive>',
    '^5': '<highfive>',
    '>_>^': '<highfive>',
    '^<_<': '<highfive>',
    '<3': '<heart>',
    '^3^': '<smile>',
    "(':": '<smile>',
    " > < ": '<smile>',
    "UvU": '<smile>',
    "uwu": '<smile>',
    'UwU': '<smile>'
}

regex_dict = {
    'URL': r"""(?xi)\b(?:(?:https?|ftp|file):\/\/|www\.|ftp\.|pic\.|twitter\.|facebook\.)(?:\([-A-Z0-9+&@#\/%=~_|$?!:;,.]*\)|[-A-Z0-9+&@#\/%=~_|$?!:;,.])*(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[A-Z0-9+&@#\/%=~_|$])""",
    'EMAIL': r"(?:^|(?<=[^\w@.)]))(?:[\w+-](?:\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(?:\.(?:[a-z]{2,})){1,3}(?:$|(?=\b))"
}


def text_preprocessing(text):
    regex = {k: re.compile(regex_dict[k]) for k, v in
             regex_dict.items()}
    for key, reg in regex.items():
        text = regex[key].sub(lambda m: " <" + key + "> ",
                              text)
    for word in emoticons.keys():
        text = text.replace(word, emoticons[word])
    text = text.lower()
    for word in contraction_mapping.keys():
        text = text.replace(word, contraction_mapping[word])
    text = re.sub(r"[\-\"`@#$%^&*(|)/~\[\]{\}:;+,._='!?]+", " ", text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', errors='ignore').decode('utf8', errors='ignore')
    text = re.sub(r'\b([b-hB-Hj-zJ-Z] )', ' ', text)
    text = re.sub(r'( [b-hB-Hj-zJ-Z])\b', ' ', text)
    text = ' '.join(text.split())
    return text