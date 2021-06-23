Visit Link :  https://transnlp.herokuapp.com/

Natural language processing (NLP) is an exciting branch of artificial intelligence (AI) that allows machines to break down and understand human language. This project includes text pre-processing techniques, machine learning techniques and Python libraries for NLP. 

There are a few packages that are used - wordcloud, textblob and gensim.

Text pre-processing techniques include:

* tokenization
* text normalization
* data cleaning. 

Web Scraping:

Web Scraping of a HTML web page to scrape the transacripts of all the comedians from "Scraps From The Loft".The notbook `get_transcripts.ipynb` crawls the Scraps From The Loft web archive for a directory of links and other data relating to transcripts from 330+ comedians' stand-up performances. Collected links are then iterated through to scrape sub-pages which contain the actual transcripts. Some cleaning and tagging is done using Regex and an IMDb api.
### The dataset that results from running `get_transcripts,ipynb` then `cleaning,ipynb` contains the following columns: 
          [title, 
          tags,  
          link, 
          name, 
          year, 
          transcript, 
          language, 
          runtime, 
          rating,
          words,
          word_count,
          f_words,
          s_words,
          diversity,
          diversity_ratio]

Cleaning Data: 

It involves removal of punctuations, making text lowe case, removal of numerical values, remove numerical values, remove common non-sensical text (/n), Tokenize text and Remove stop words.

Explore Data: 

In this process, I found out the most common words used by some specific comedian. I generated word clouds. This was done to get an idea of what makes that comedian the top one/ trending one? 
I also figured out the number of unique words.
I also figured out the bad words used.

Sentiment Analysis :

1. TextBlob Module: Linguistic researchers have labeled the sentiment of words based on their domain expertise. Sentiment of words can vary based on where it is in a sentence. The TextBlob module allows us to take advantage of these labels.

2. Sentiment Labels: Each word in a corpus is labeled in terms of polarity and subjectivity (there are more labels as well, but we're going to ignore them for now). A corpus' sentiment is the average of these.

* Polarity: How positive or negative a word is. -1 is very negative. +1 is very positive.

* Subjectivity: How subjective, or opinionated a word is. 0 is fact. +1 is very much an opinion.

Topic Modelling:

The ultimate goal of topic modeling is to find various topics that are present in our corpus. Each document in the corpus will be made up of at least one topic, if not multiple topics.. In this project, I covered Latent Dirichlet Allocation (LDA), which is one of many topic modeling techniques. It was specifically designed for text data. Result is a vector for each transcript that indicates probabilities for the presence of different topics.

Text Generation:

I used Markov chains for text generation. Think about every word in a corpus as a state. We can make a simple assumption that the next word is only dependent on the previous word - which is the basic assumption of a Markov chain.

Predict IMDb rating: 
Use the LDA vector along with a handmade binary target feature `rating_type` (1 for above average and 0 for below average IMDb rating) to train an ensemble classifier.


The script will work to grab as many transcripts as are available at runtime... when the site adds more performances to their archive, the resulting csv output will grow.
