import Stemmer

# Create a new Lithuanian stemmer.
lithuanian_stemmer = Stemmer.Stemmer('lithuanian')

# Stem a word.
stemmed_words = lithuanian_stemmer.stemWords(['universitetas', 'universiteto'])

print(stemmed_words)  # Output will be the stemmed versions of the provided words.