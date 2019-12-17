from __future__ import unicode_literals, print_function
from gensim.summarization import summarize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords

ttt = nltk.tokenize.TextTilingTokenizer()
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
stop = set(stopwords.words("english"))

def summarize_text(text):
    """ Returns a summary of the raw text provided
    """
    return summarize(text)

def get_topics(text, no_topics=3, no_top_words=5):
    """ Returns no_topics topics each consisting of no_top_words words
    """
    data = "".join(e if e.isalnum() else " " for e in text).split()
    data = [d for d in data if d not in stop]
    tf = tf_vectorizer.fit_transform(data)
    tf_feature_names = tf_vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_topics=no_topics, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    return [" ".join([tf_feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]) for topic in lda.components_]

def segment(text):
    """ Takes a text composed of various paragraphs and tries to intelligently segment it into various subparts
    """
    return text.split("\n")


# data = "Thomas A. Anderson is a man living two lives. By day he is an average computer programmer and by night a hacker known as Neo. Neo has always questioned his reality, but the truth is far beyond his imagination. Neo finds himself targeted by the police when he is contacted by Morpheus, a legendary computer hacker branded a terrorist by the government. Morpheus awakens Neo to the real world, a ravaged wasteland where most of humanity have been captured by a race of machines that live off of the humans' body heat and electrochemical energy and who imprison their minds within an artificial reality known as the Matrix. As a rebel against the machines, Neo must return to the Matrix and confront the agents: super-powerful computer programs devoted to snuffing out Neo and the entire human rebellion."
# data = "".join(e if e.isalnum() else " " for e in data).split()
# # print(get_topics(data))
# sample = """we have done hackathon, let me tell you a story. why because we couldnt find any teammates.thats where we came up with the idea of startospace. for highschool students.and very soonComing to the existing market problems, first of all what problems do we face in the university. we dont usually get to decide the teammates. for example, if I want to participate in this hackathon. My ony source of finding teammates is my friends, or my contacts. What If I want to make an app but I dont have any friends who are app developers, this is where starterspace links as a platform to find teammates for startups or even course to course

# How starter space is actually implemented, Starter space is divided into three states. first one we call it as which is the umm online projects phase. the projects space is the space where basically the students can use the different tools which we have for example the which we developed using the azure which actually helps the students to practice computer science and students can easily find the events So there are many market trends like who actually connect the team members they actually connect the team members and can find the team members you can even join the team members but we are unique because it is not limited to the competitions that the company organizes here we can, for example I can create a competition and I can even post an event on say get the teams for it executive summary, less time data

# we can collect some forms of data which the universities can use to track how the students participate in competitions and which are not just limited to the universityat starter space we have both an app and a website, im going to show you. highschool students. 
# so after choosing the you enter your username and password let me explain, the pin page is the universities notice board and anybody can post about any thing you have. now we move to the page. and moving on to the play. the play page is where you can find the list of competitions globally, locally and even hong kong wide. why not just check hong kong. you find details about the hackathon and the most interesting thing about the hackathon."""

# summary = summarize_text(sample)
# topics_whole = get_topics(sample)
# segmented_text = segment(sample)
# topics = [get_topics(segment) for segment in segmented_text if len(segment)>10]
# print(summary, "\n")
# print(topics_whole, "\n")
# print(topics)