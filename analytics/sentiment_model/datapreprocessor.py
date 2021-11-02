import pickle
import pandas as pd
import numpy as np
import re
import itertools
import string
import nltk
import sys

nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

from datetime import datetime
from pandas.tseries.offsets import MonthEnd
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification

lemmatizer = WordNetLemmatizer()
finbert = BertForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone", num_labels=3
)
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

# Filtered words
economic_growth = ["economic", "economy", "growth", "slowdown", "outlook"]
employment_growth = [
    "labor",
    "labour",
    "unemployment",
    "employment",
    "job",
    "staff",
    "wage",
    "employ",
]
inflation = [
    "price",
    "prices",
    "inflation",
    "stagflation",
    "disinflation",
    "spending",
    "spendings",
    "cost",
    "inflationary",
]
filter_words = list(itertools.chain(economic_growth, employment_growth, inflation))
us_words = ["us", "US", "United States", "united states", "america", "americans"]

# Excluded and stop words
excluded = ["no", "not"]
stop = stopwords.words("english")


class DataPreprocessor:
    def __init__(self, data, batch_id):
        self.batch_id = batch_id
        self.data = self.preprocess(data)  # dictionary of dfs

    def clean_text(self, text):
        """
        Clean text, remove stop words and punctuations
        """
        text = " " + text + " "
        text = text.replace("-", " ")  # remove hyphen
        text = text.replace("\n", " ")
        text = text.replace("\n[SECTION]\n", "<stop>")
        text = text.replace("[SECTION]", "<stop>")
        text = text.replace("\r", "")
        text = text.replace("'s", "")  # remove 's in python
        text = text.replace(
            "[^a-zA-Z]+", " "
        )  # remove any character that is not a-z OR A-Z
        text = re.sub(prefixes, "\\1<prd>", text)
        text = re.sub(websites, "<prd>\\1", text)
        if "Ph.D" in text:
            text = text.replace("Ph.D.", "Ph<prd>D<prd>")
        text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
        text = re.sub(
            alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]",
            "\\1<prd>\\2<prd>\\3<prd>",
            text,
        )
        text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
        text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
        text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
        if "”" in text:
            text = text.replace(".”", "”.")
        if '"' in text:
            text = text.replace('."', '".')
        if "!" in text:
            text = text.replace('!"', '"!')
        if "?" in text:
            text = text.replace('?"', '"?')
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        text = text.lower()  # lower

        return text

    def split_into_sentences(self, text):
        """
        Split sentences and check for whitespaces, single letter removal
        Ensure spacing are all single spacing
        """

        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]  # remove whitespaces
        sentences = [
            re.sub("(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)", "", s) for s in sentences
        ]  # remove single letter string
        sentences = [
            " ".join(s.split()) for s in sentences
        ]  # ensure all spacing is single spacing

        return sentences

    def filter_articles(self, sentences):
        """
        Filter relevant news articles
        """

        new_sentences = []
        if len(sentences) == 0:
            return new_sentences

        if any(s in sentences[0].lower() for s in filter_words) and any(
            s in sentences[0].lower() for s in us_words
        ):
            # remove punctuations
            sentence = "".join([c for c in sentences[0] if c not in string.punctuation])

            # remove numbers
            sentence = re.sub(r"[^a-zA-z.,!?/:;\"\'\s]", "", sentence)

            # remove single letters
            sentence = re.sub(
                "(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)", "", sentence
            )  # remove single letter string
            new_sentences.append(sentence)

        return new_sentences

    def filter_relevant(self, sentences):  # take in list sentence as input
        """
        for machine learning classification, only filter relevant keywords with last text data cleaning
        """
        new_sentences = []

        for sentence in sentences:

            if any(s in sentence.lower() for s in filter_words):

                # remove punctuations
                sentence = "".join([c for c in sentence if c not in string.punctuation])

                # remove numbers
                sentence = re.sub(r"[^a-zA-z.,!?/:;\"\'\s]", "", sentence)

                # remove single letters
                sentence = re.sub(
                    "(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)", "", sentence
                )  # remove single letter string

                new_sentences.append(sentence)

        if len(new_sentences) == 0:  # revert to no filter usage

            for sentence in sentences:

                # remove punctuations
                sentence = "".join([c for c in sentence if c not in string.punctuation])

                # remove numbers
                sentence = re.sub(r"[^a-zA-z.,!?/:;\"\'\s]", "", sentence)

                # remove single letters
                sentence = re.sub(
                    "(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)", "", sentence
                )  # remove single letter string

                if len(sentence) < 1:  # means empty string
                    continue

                else:
                    new_sentences.append(sentence)

        return new_sentences

    # FOR DICTIONARY-BASED CLASSIFICATION
    def remove_non_text(self, sentences):  # take in list sentence as input
        """
        for dictionary-based method, with last text data cleaning
        """

        new_sentences = []

        for sentence in sentences:

            # remove punctuations
            sentence = "".join([c for c in sentence if c not in string.punctuation])

            # remove numbers
            sentence = re.sub(r"[^a-zA-z.,!?/:;\"\'\s]", "", sentence)

            # remove single letters
            sentence = re.sub(
                "(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)", "", sentence
            )  # remove single letter string

            if len(sentence) < 1:  # means empty string
                continue

            else:
                new_sentences.append(sentence)

        return new_sentences

    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }

        return tag_dict.get(tag, wordnet.NOUN)

    def lemmatize_list(self, sentences):
        new_sentences = []

        for sentence in sentences:

            if len(sentence) == 0:
                continue

            lemmatized_list = [
                lemmatizer.lemmatize(w, self.get_wordnet_pos(w))
                for w in nltk.word_tokenize(sentence)
            ]
            lemmatized_word = " ".join(lemmatized_list)
            new_sentences.append(lemmatized_word)

        return new_sentences

    def filter_neutral(self, sentences):  # take in list sentence as input
        """
        Transfer learning using FinBERT to filter neutral sentiments
        Note: sentences cannot be an empty list, else function will not work
        """

        if len(sentences) == 0:
            print("Empty List Alert")
        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        outputs = finbert(**inputs)[0]

        new_sentences = []

        for idx, sent in enumerate(sentences):
            if np.argmax(outputs.detach().numpy()[idx]) != 0:  # not neutral
                new_sentences.append(sent)

        return new_sentences

    def preprocess(self, data):
        """
        Data Cleaning and Preprocessing
        """
        for k, df in data.items():

            if k == "historical":
                continue

            if k == "statements" or k == "minutes" or k == "news":
                print(f"===== {k} preprocessing =====".title())

                if k == "news":  # extra preprocessing step for news data

                    # Obtain relevant rows for analysis
                    df.dropna(
                        subset=["article snippet"], inplace=True
                    )  # remove na values
                    df = df[(df.material_type == ("News"))]  # filter only news
                    df.rename(columns={"article snippet": "contents"}, inplace=True)
                    df.reset_index(inplace=True, drop=True)
                    df[
                        "title"
                    ] = None  # TODO: add better functionality to address this use case

                # replace \n and \r with empty string, and remove trailing whitespaces
                df["text"] = df["contents"].apply(
                    lambda x: x.replace("\n", " ").replace("\r", " ").strip()
                )
                df.drop(columns=["title"], axis=1, inplace=True)

                df["contents"] = df["contents"].apply(lambda x: self.clean_text(x))

                # Remove stopwords
                df["contentClean"] = df["contents"].apply(
                    lambda x: " ".join(
                        x for x in x.split() if (x not in stop) or (x in excluded)
                    )
                )

                # Convert to sentences
                df["sentences"] = df["contentClean"].apply(
                    lambda x: self.split_into_sentences(x)
                )

                # Filter relevant articles for news
                if k == "news":
                    print(f"===== {k} filtering relevant articles =====".title())
                    df["filterSentences"] = df["sentences"].apply(
                        lambda x: self.filter_articles(x)
                    )
                    df = df[
                        (df["filterSentences"].str.len() != 0)
                    ]  # remove empty list that are irrelevant articles
                    print(f"===== {k} finish filtering relevant articles =====".title())

                else:

                    df["filterSentences"] = df["sentences"].apply(
                        lambda x: self.filter_relevant(x)
                    )  # for machine learning model

                df["filterSentencesDB"] = df["sentences"].apply(
                    lambda x: self.remove_non_text(x)
                )  # for dictionary-based

                # For machine learning classification
                print(f"===== {k} lemmatize relevant articles =====".title())
                df["lemmatizedSentences"] = df["filterSentences"].apply(
                    lambda x: self.lemmatize_list(x)
                )

                df["lemmatizedSentencesDB"] = df["filterSentencesDB"].apply(
                    lambda x: self.lemmatize_list(x)
                )
                print(f"===== {k} lemmatize relevant articles done =====".title())

                if k == "news":

                    # aggregate by monthly basis
                    df["date"] = pd.to_datetime(df["date"]) + MonthEnd(0)
                    df = df[
                        (df["lemmatizedSentences"].str.len() != 0)
                    ]  # remove empty list
                    df = (
                        df.groupby("date")["lemmatizedSentences"]
                        .apply(list)
                        .reset_index(name="lemmatizedSentences")
                    )
                    df["lemmatizedSentences"] = df["lemmatizedSentences"].apply(
                        lambda x: list(itertools.chain.from_iterable(x))
                    )

                print(f"===== {k} filter out neutral articles =====".title())
                df["lemmatizedSentences"] = df["lemmatizedSentences"].apply(
                    lambda x: self.filter_neutral(x) if len(x) != 0 else x
                )
                print(f"===== {k} filter out neutral articles done =====".title())
                # Save df as pickle
                self.save_df(k, df)

                # Replace the df with preprocesses ones
                data[k] = df

        return data

    def save_df(self, name, df):
        """
        Save df to pickle
        """

        rename_dict = {"statements": "st", "minutes": "mins", "news": "news"}

        df.to_pickle(
            f"../data/db/pickle/preprocess/{self.batch_id}_{rename_dict[name]}_df.pickle"
        )
