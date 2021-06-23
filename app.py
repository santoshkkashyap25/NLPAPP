# importing Required Libraries

import streamlit as st

import pickle
import joblib

import math
import random
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# from wordcloud import WordCloud

from PIL import Image

from collections import defaultdict

from textblob import TextBlob

import warnings
warnings.filterwarnings('ignore')



pd.set_option('display.max_colwidth', 2000)
sns.set(color_codes=True)


def main():
	# Image.open('nlpimage.png').convert('RGB').save('nlp.jpeg')
	icon = Image.open("NLP-image.jpg")

	# setting initial configurations
	# streamlit config show
	st.set_page_config(
	    page_title="Real Time Transcription NLP Analysis",
	 	page_icon=icon,
	    layout="centered",
	    initial_sidebar_state="collapsed")


	
	def Load_data():
		df = pd.read_csv("frame2_cleaned.csv")
		return df



	# defines homepage
	def home():
		st.header("Real Time Transcript NLP Analysis")
		st.markdown(" Machine Learning & Natural Language Processing Project")
		st.write("Please select a page on the left sidebar")
		st.image(icon)



	# Showing the cleaned Dataset
	def show_data():
		df= Load_data()
		data = st.selectbox(
	        "Select a option to see",
	        [
	            "Show Full Dataset",
	        	"Show Dataset Head",
	        	"Show Dataset Tail",
	        	"Show Comedian Names",
	        	"Show Dataset Description"
	        	
	        ],
	    )
		
		
		if data == "Show Full Dataset":
			st.write(df)

		if data == "Show Dataset Head":
			st.write(df.head())

		if data == "Show Dataset Tail":
			st.write(df.tail())

		if data == "Show Comedian Names":
			st.write(df['Names'].unique())

		if data == "Show Dataset Description":
			st.table(df.describe())



	# defines the show transcripts fuction
	def transcripts():
		st.write("""Scraped the web then preprocessed and did some cleaning and then saved to csv for 
					further analysis. Luckily, there are wonderful people online that keep track 
					of stand up routine transcripts. Scraps From The Loft makes them available for
					non-profit and educational purposes.""")
		
		st.markdown("### Enter the Serial No. of the Transcript to see it.")
		st.text("Please check Serial No. here")
		df= Load_data()
		if st.button("SHOW DATASET"):
			st.dataframe(df)
		load_transcripts = joblib.load('transcripts_joblib')
		num = st.number_input('Enter Number',min_value=0, max_value=len(df)-1, value=0)
		num = int(num)
		st.write(load_transcripts[num])



	def kde():
		frame=pd.read_csv("frame2_cleaned.csv")
		plot = st.selectbox(
	        "Select a Plot for visualization",
	        [
	            "Transcript Character Count KDE",
	        	"Runtime KDE",
	        	"IMDb Rating KDE",
	        	"F-Words Count KDE",
	        	"S-Words Count KDE",
	        	"Word Diversity KDE",
	        	"Diversity / Total words KDE"
	        ],
	    )
		

		if plot == "Transcript Character Count KDE":
			x = [len(x) for x in frame.Transcript]
			fig=plt.figure()
			sns.kdeplot(x, shade=True, color="b")
			plt.title('Transcript Character Count KDE')
			st.pyplot(fig)
			mean = np.array(x).mean()
			sd = np.array(x).std()
			st.write((f'Mean: {mean}'))
			st.write((f'Standard Deviation: {sd}'))

		
		if plot == "Runtime KDE":
			x = []
			count = 0
			for i in frame.runtime:
				if (i > 0):
				    count += 1
				    x.append(int(i))
			fig =plt.figure()
			sns.kdeplot(x, shade=True, color="r")   
			plt.title('Runtime KDE')
			plt.xlabel('minutes')
			st.pyplot(fig)
			mean = np.array(x).mean()
			sd = np.array(x).std()
			st.write((f'Mean: {mean}'))
			st.write((f'SD: {sd}'))

		
		if plot == "IMDb Rating KDE":
			x = []
			count = 0
			for i in frame.rating:
			    if (i > 0):
			        count += 1
			        x.append(i)
			fig = plt.figure()
			sns.kdeplot(x, shade=True, color="g")   
			plt.title('IMDb Rating KDE')
			st.pyplot(fig)
			mean = np.array(x).mean()
			sd = np.array(x).std()
			st.write((f'Mean: {mean}'))
			st.write((f'SD: {sd}'))

		
		if plot == "F-Words Count KDE":
			fig = plt.figure()
			sns.kdeplot(frame.f_words, shade=True, color="r")
			plt.title('F-Words Count KDE')
			st.pyplot(fig)
			mean = frame.f_words.mean()
			sd = frame.f_words.std()
			st.write((f'Mean: {mean}'))
			st.write((f'SD: {sd}'))


		if plot == "S-Words Count KDE":
			fig = plt.figure()
			sns.kdeplot(frame.s_words, shade=True, color="r")
			plt.title('S-Words Count KDE')
			st.pyplot(fig)
			mean = frame.s_words.mean()
			sd = frame.s_words.std()
			st.write((f'Mean: {mean}'))
			st.write((f'SD: {sd}'))

		
		if plot == "Word Diversity KDE":
			fig = plt.figure()
			sns.kdeplot(frame.diversity, shade=True, color="purple")
			plt.title('Word Diversity KDE')
			st.pyplot(fig)
			mean = frame.diversity.mean()
			sd = frame.diversity.std()
			st.write((f'Mean: {mean}'))
			st.write((f'SD: {sd}'))

		
		if plot == "Diversity / Total words KDE":
			fig = plt.figure()
			sns.kdeplot(frame.diversity_ratio, shade=True, color="g")
			plt.title('Diversity / Total words KDE')
			st.pyplot(fig)
			mean = frame.diversity_ratio.mean()
			sd = frame.diversity_ratio.std()
			st.write((f'Mean: {mean}'))
			st.write((f'SD: {sd}'))



	def rating():
		frame= pd.read_csv("frame_rating.csv")
		st.write("""Given a 1 for any rating above the mean, and a 0 otherwise. 
					This will be our target for a classification task .""")
		
		st.write("""High rating (> mean) And Low rating (< mean)]""")
		
		frame['rating_type'] = frame.rating.apply(lambda x: 1 if x >= frame.rating.mean() else 0)
		title='Counts of specials with higher or lower than average ratings'
		fig = plt.figure()
		sns.countplot(x='rating_type', data=frame)
		plt.title("Counts of specials with higher or lower than average ratings")
		st.pyplot(fig)


	def pair():
		frame=pd.read_csv("frame2_cleaned.csv")
		st.write("Pairplot visualization to discover correlations")
		fig = sns.pairplot(frame[['diversity_ratio', 'diversity', 'word_count', 'runtime', 'rating', 'rating_type']])
		st.pyplot(fig)




	#def wordcloud():
	#	df = Load_data()
	#	st.markdown("### Enter the Serial No. of the Transcript to see it's Wordcloud.")
	#	st.text("Please check Serial No. here")
	#	if st.button("SHOW DATASET"):
	#		st.dataframe(df)
	#	num = st.number_input('Enter Serial Number',min_value=0, max_value=len(df)-1, value=0)
	#	num = int(num)
	#	st.write((df.title[num]))
	#	wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='midnightblue')
	#	wordcloud.generate(' '.join(df.words[num]))
	#	wordcloud.to_image()




	# for the sentiment Analysis
	def sentiments():
		senti = st.selectbox(
	        "Select a Option to Explore",
	        [
	            "Show Indiviadual Sentiments",
	        	"Show Sentiment Plot",
	        	"Show Sentiment Plot Over Routine",
	        ],
	    ) 

		if senti == "Show Indiviadual Sentiments": 
			df= Load_data()
			st.subheader("Enter the comedian name you'd like to analyze sentiments.")
			if st.button("Show Comedian Names"):
				st.write(df['Names'].unique())
			text = st.text_input('Enter Name')
			i=df[df["Names"]==text]
			st.write(i)
			dff = pd.DataFrame()
			dff['S.No'] = i['S No.'].copy()
			dff['Name'] = i['Names'].copy()
			dff['Title Name'] = i['title'].copy()
			pol = lambda x: TextBlob(x).sentiment.polarity
			sub = lambda x: TextBlob(x).sentiment.subjectivity
			dff['polarity'] = i['Transcript'].apply(pol)
			dff['subjectivity'] = i['Transcript'].apply(sub)
			st.write("Sentiments Analysis says")
			st.table(dff)

		if senti == "Show Sentiment Plot":
			df=pd.read_csv("frame_senti.csv")
			st.write("To get Sentiment Plot pick a forward Subset of the Dataset")
			st.markdown("## Note-Please Pick a subset of atmost range of 20 to get clear Visualization.")
			st.markdown("### Enter the Serial Numbers of the Transcript to visualize Plot.")
			st.text("Please check Serial Numbers here")
			if st.button("SHOW DATASET"):
				st.dataframe(df)
			num1 = st.number_input('Enter Starting Serial Number',min_value=0, max_value=len(df)-1, value=0)
			num1 = int(num1)
			num2 = st.number_input('Enter Ending Serial Number',min_value=0, max_value=len(df)-1, value=0)
			num2 = int(num2)
			st.write("Please press Enter")
			data= df[num1:num2]
			st.subheader("Sentiment Plot")
			fig= plt.figure()
			plt.rcParams['figure.figsize'] = [15, 15]
			for index, comedian in enumerate(data['S No.']):
			    x = data.polarity.loc[comedian]
			    y = data.subjectivity.loc[comedian]
			    plt.scatter(x, y, color='red')
			    plt.text(x+.001, y+.001, data['Names'][index+num1], fontsize=15)
			plt.title('Sentiment Analysis Plot', fontsize=30)
			plt.xlabel('<-- Negative -------- Positive -->', fontsize=20)
			plt.ylabel('<-- Facts -------- Opinions -->', fontsize=20)
			st.pyplot(fig)


		if senti == "Show Sentiment Plot Over Routine":
			df=pd.read_csv("frame_senti.csv")
			st.write("To get Sentiment Plot over Routine pick a forward Subset of the Dataset")
			st.markdown("## Note-Please Pick a subset of atmost range of 20 to get clear Visualization.")
			st.markdown("### Enter the Serial Numbers of the Transcript to visualize Plot.")
			st.text("Please check Serial Numbers here")
			if st.button("SHOW DATASET"):
				st.dataframe(df)
			num1 = st.number_input('Enter Starting Serial Number',min_value=0, max_value=len(df)-1, value=0)
			num1 = int(num1)
			num2 = st.number_input('Enter Ending Serial Number',min_value=0, max_value=len(df)-1, value=0)
			num2 = int(num2)
			data= df[num1:num2]
			st.write("Please press Enter")
			st.subheader("Sentiment Plot Over Routine")
			
			def split_text(text, n=10):
			    length = len(text)
			    size = math.floor(length / n)
			    start = np.arange(0, length, size)
			    split_list = []
			    for piece in range(n):
			        split_list.append(text[start[piece]:start[piece]+size])
			    return split_list

			list_pieces = []
			for t in data.Transcript:
			    split = split_text(t)
			    list_pieces.append(split)
    
			polarity_transcript = []
			for lp in list_pieces:
			    polarity_piece = []
			    for p in lp:
			        polarity_piece.append(TextBlob(p).sentiment.polarity)
			    polarity_transcript.append(polarity_piece)
    		

			fig=plt.figure()
			plt.rcParams['figure.figsize'] = [25, 25]
			for index, comedian in enumerate(data['S No.']):    
			    plt.subplot(4,5 , index+1)
			    plt.plot(polarity_transcript[index])
			    plt.plot(np.arange(0,10), np.zeros(10))
			    plt.title(data['Names'][index+num1])
			    plt.ylim(ymin=-.2, ymax=.3)
			st.pyplot(fig)
    




	# for the text generation
	def markov_chain(text):
	    words = text.split(' ')
	    m_dict = defaultdict(list)
	    for current_word, next_word in zip(words[0:-1], words[1:]):
	        m_dict[current_word].append(next_word)
	    m_dict = dict(m_dict)
	    return m_dict



	def generate_sentence(chain, count=100):
	    word1 = random.choice(list(chain.keys()))
	    sentence = word1.capitalize()
	    for i in range(count-1):
	        word2 = random.choice(chain[word1])
	        word1 = word2
	        sentence += ' ' + word2
	    sentence += '.'
	    return(sentence)



	def text():
		data = pd.read_pickle('corpus.pkl')
		st.markdown("### Enter the Serial No. of the Transcript for Text Generation.")
		st.text("Please check Serial No. here-")
		df= Load_data()
		if st.checkbox("SHOW DATASET"):
			st.dataframe(df)
		num = st.number_input('Enter Number',min_value=0, max_value=len(df)-1, value=0)
		num = int(num)
		st.header(df.title[num])
		com_text = data.Transcript.loc[num]
		com_dict = markov_chain(com_text)
		st.write("The Generated Text-")
		st.write(generate_sentence(com_dict))
		st.header("NOTE-")
		st.markdown(""" ##### The generated text looks like English with only a few spelling errors though it doesn't really make much sense. There are grammar and syntax errors everywhere but this is partially to be expected given that the source text is composed of transcripts from spoken stand-up comedy routines. Unfortunately, due to computational and time constraints, the full corpus was not used.""")



	# defines topic modelling
	def topic():
		dff = pd.read_pickle('topic_modelLDA.pkl')
		st.markdown("### Enter the Serial No. of the Transcript to see Topics Probabilities.")
		st.text("Please check Serial No. here-")
		df= Load_data()
		if st.checkbox("SHOW DATASET"):
			st.dataframe(df)
		num = st.number_input('Enter Number',min_value=0, max_value=len(df)-1, value=0)
		num = int(num)
		st.write(dff.title[num])
		st.table(dff.loc[num][17:24])

	def meanprob():
		st.subheader("Mean Topic Probabilities Across Dataset")
		df = pd.read_pickle('topic_modelLDA.pkl')
		fig= plt.figure()
		topics = ['Crimes', 'Big Picture', 'UK', 'relationships', 'Foods', 'politics', 'clean']
		sns.barplot(x=df[topics].mean().index, y=df[topics].loc[df.rating_type == 1].mean())
		plt.title('Mean Topic Probabilities Across The Entire Dataset')
		plt.xlabel('Topics', fontsize=20)
		plt.ylabel('Mean Percentage per Transcript', fontsize=20)			
		plt.ylim(0, 0.5)
		st.pyplot(fig)



	page = st.sidebar.selectbox(
	        "Select a Page",
	        [
	            "Homepage",
	            "Dataset",
	            "Scraped Transcripts",
	            "Data Visualization",
	            "Sentiment Analysis",
	            "Text Generator",
	            "Topic Modelling",
	            "About Project"
	        ],
	    )



	if page == "Homepage":
		home()


	elif page == "Dataset":
		st.markdown("## Real Time Transcript NLP Analysis")
		if st.checkbox("Explore Dataset"):
			show_data()


	elif page == "Scraped Transcripts":
		st.markdown("## Real Time Transcript NLP Analysis")
		if st.checkbox("Show Transcripts"):
			transcripts()


	elif page == "Data Visualization":
		st.markdown("## Real Time Transcript NLP Analysis")
		st.write("""KDE plots are used throughout to make sure everything 
					looks right. Quickly viewing a simple distribution can 
					be a great indicator that the code and data are performing
					as expected.""")
		if st.checkbox("Show KDE Plots"):
			kde()

		elif st.checkbox("Show Rating Type Count Plot"):
			rating()

		elif st.checkbox("Show Pairplot Visualization"):
			pair()


	elif page == "Sentiment Analysis":
		st.markdown("## Real Time Transcript NLP Analysis")
		st.write("""A corpus' sentiment is the average of these.""")
		st.write("""Polarity: How positive or negative a word is.
					-1 is very negative. +1 is very positive.""")
		st.write("""Subjectivity: How subjective, or opinionated a word is.
					0 is fact. +1 is very much an opinion.""")
		if st.checkbox("Do Sentiment Analysis"):
			sentiments()


	elif page == "Text Generator":
		st.markdown("## Real Time Transcript NLP Analysis")
		if st.checkbox("Do Text Generation"):
			text()


	elif page == "Topic Modelling":
		st.markdown("## Real Time Transcript NLP Analysis")
		
		if st.checkbox("SHOW TOPIC PROBABILITIES"):
			topic()

		elif st.checkbox("SHOW MEAN TOPICS PROBABILITIES PLOT"):
			meanprob()


	elif page == "About Project":
		st.header("Natural Language Processing Project")
		
		st.write("""Natural language processing (NLP) is an exciting branch of 
					artificial intelligence (AI) that allows machines to break 
					down and understand human language. This project includes 
					text pre-processing techniques, machine learning techniques 
					and Python libraries for NLP.""")
		
		st.write("""The flow of the project is described as below- """)
		
		st.write("""Web Scraping of a HTML web page to scrape the transcripts 
					of all the comedians and then processing it into dataset with 
					other details.""")
		
		st.write("""There are a few packages that are used - wordcloud, textblob 
					and gensim.Text pre-processing techniques include tokenization,
					text normalization, data cleaning.""")
		
		st.write("""Cleaning Data involves removal of punctuations, making text 
					lower case, removal of numerical values, remove numerical 
					values, remove common non-sensical text (/n), Tokenize text 
					and Remove stop words.""")
		
		st.write("""Explore Data- In this process, I found out the most common 
					words used by some specific comedian. I generated word clouds.
					This was done to get an idea of what makes that comedian the 
					top one/ trending one?""")
		
		st.write("""Sentiment Analysis- Linguistic researchers have labeled the 
					sentiment of words based on their domain expertise. Sentiment 
					of words can vary based on where it is in a sentence. The 
					TextBlob module allows us to take advantage of these labels.""")
		
		st.write("""Polarity: How positive or negative a word is. -1 is very 
					negative. +1 is very positive.""")
		
		st.write("""Subjectivity: How subjective,or opinionated a word is. 0 is fact. +1 is very much an opinion.
					""")
		
		st.write("""Topic Modelling - The ultimate goal of topic modeling is to 
					find various topics that are present in our corpus. Each 
					document in the corpus will be made up of at least one topic, 
					if not multiple topics.. In this project, I covered Latent 
					Dirichlet Allocation (LDA), which is one of many topic modeling
					techniques. It was specifically designed for text data. Result 
					is a vector for each transcript that indicates probabilities 
					for the presence of different topics.""")
		
		st.write("""Text Generation - I used Markov chains for text generation. 
					Think about every word in a corpus as a state. We can make 
					a simple assumption that the next word is only dependent on
					the previous word - which is the basic assumption of a 
					Markov chain.""")
		
		st.write("""Predict IMDb rating - Use the LDA vector along with a 
					handmade binary target feature "rating_type" 
					(1 for above average and 0 for below average IMDb rating) 
					to train an ensemble classifier.""")




if __name__ == '__main__':
    main()