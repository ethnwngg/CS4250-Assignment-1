#-------------------------------------------------------------------------
# AUTHOR: Ethan Wong
# FILENAME: indexing.py
# SPECIFICATION: Outputting the tf-idf document-term matrix as specified in question 7
# FOR: CS 4250 - Assignment #1
# TIME SPENT: 6 hours
#-----------------------------------------------------------*/

#Importing some Python libraries
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

documents = []

#Reading the data in a csv file
with open('collection.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
         if i > 0:  # skipping the header
            documents.append (row[0])

#Conducting stopword removal for pronouns/conjunctions. Hint: use a set to define your stopwords.
#--> add your Python code here
stopWords = {"i", "and", "she", "her", "they", "their"}
stopWords = {word.lower() for word in stopWords}

#Conducting stemming. Hint: use a dictionary to map word variations to their stem.
#--> add your Python code here
stemming = {"loves": "love", "cats": "cat", "dogs": "dog"}

filteredDocuments = []
for doc in documents:
        words = doc.lower().split()

        filtered_words = []
        for word in words:
             if word not in stopWords:
                  stemmed_word = stemming.get(word, word)
                  filtered_words.append(stemmed_word)
        filteredDocuments.append(" ".join(filtered_words))

#Identifying the index terms.
#--> add your Python code here
terms = []
for document in filteredDocuments:
    for word in document.split():
         if word not in terms:
              terms.append(word)

#Building the document-term matrix by using the tf-idf weights.
#--> add your Python code here
docTermMatrix = []

vectorizer = TfidfVectorizer()
docTermMatrix = vectorizer.fit_transform(filteredDocuments).toarray()
df_docTermMatrix = pd.DataFrame(docTermMatrix, columns=vectorizer.get_feature_names_out())

#Printing the document-term matrix.
#--> add your Python code here

print(df_docTermMatrix)
