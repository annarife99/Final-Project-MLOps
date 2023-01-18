import re
import string

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


def convert_to_lower(text):
    return text.lower()


def remove_emojis(text):
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)  # remove links and mentions
    text = re.sub(r"<.*?>", "", text)

    wierd_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u200d"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\u3030"
        "\ufe0f"
        "\u2069"
        "\u2066"
        # u"\u200c"
        "\u2068" "\u2067" "]+",
        flags=re.UNICODE,
    )

    return wierd_pattern.sub(r"", text)


def remove_numbers(text):
    number_pattern = r"\d+"
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number


def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_stopwords(text):
    removed = []
    stop_words = list(stopwords.words("english"))
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            removed.append(tokens[i])
    return " ".join(removed)


def remove_extra_white_spaces(text):
    single_char_pattern = r"\s+[a-zA-Z]\s+"
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc


def preprocessText(text):
    return remove_extra_white_spaces(
        remove_stopwords(remove_punctuation(remove_numbers(remove_emojis(convert_to_lower(text)))))
    )


def preprocessBatch(batch):
    new_list = []
    for i in batch["text"]:
        new_list.append(
            remove_extra_white_spaces(
                remove_stopwords(
                    remove_punctuation(remove_numbers(remove_emojis(convert_to_lower(i))))
                )
            )
        )
    batch["text"] = new_list
    return batch
