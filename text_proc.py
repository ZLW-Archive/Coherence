import re
from nltk.tokenize import TweetTokenizer

tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)

def sentence_split(sentence):
    return tweet_tokenizer.tokenize(sentence)

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        result = " ".join(["<hashtag>"] + hashtag_body.split(r"(?=[A-Z])"))
    return result

def utterance_process(text):
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    text = text.lower()

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=re.MULTILINE | re.DOTALL)

    text = re_sub(r"/n", " ")
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/", " / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lol>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sad>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutral>")
    text = re_sub(r"<3", "<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    # replace person and location
    text = text.replace("person_<number>", "person")
    text = text.replace("location_<number>", "location")

    text = re_sub(r"(\w)\1{2,}(\S*)\b", r"\1\2 <repeat>")
    try:
        # remove unicode
        text = re.sub(r"\u0092|\x92", "'", text)
        text = text.encode("utf-8").decode("ascii", "ignore")
    except:
        pass
    # split with punctuations
    # text = re_sub(r"([^A-Za-z0-9\_]+)", r" \1 ")
    text = sentence_split(text)

    return text