#Text Preprocessing and Vectorization
##Data Source
The dataset used for this project is the IMDB Review Dataset, obtained from Kaggle. It contains 50,000 labeled movie reviews, evenly split between positive and negative sentiment categories. The dataset includes 2 columns: review containing the raw text of the review and sentiment containing the label (“positive” or “negative”). 

##Preprocessing and Text Cleaning
Raw text from the review column was unstructured and contained noise that could interfere with model performance. A cleaning pipeline was implemented to standardize and prepare the data. The following steps were applied to every review:
1. Convert to lowercase- All characters were converted to lowercase so that words such as “Good” and “good” are treated as the same term by the model.
2. Remove HTML tags- Many reviews contained leftover HTML markup such as <br/> line break tags; these were stripped out entirely using regular expressions.
3. Remove Non-Alphabetical Characters- Punctuation, numbers, symbols, and emojis were removed. Only letters and spaces were retained.
4. Trim Extra Whitespace- Multiple consecutive spaces were collapsed into single spaces and leading/ trailing spaces were removed.
5. Remove Stop Words- Common words like that carry little sentiment meaning were filtered out. This includes words such as: the, is, a, it, and, this, that, was, I, you, he, she, they, at, on, by, with.
6. Filter Short Words- Any remaining word with fewer than 3 characters was removed to reduce noise.
###Example Transformation:
Before Cleaning	After Cleaning
“This movie was absolutely wonderful!!! I loved every moment of it!!!”	“movie absolutely wonderful loved every moment”
“The acting was terrible and plot was boring”	“acting terrible plot boring”

##Converting Text to Numerical Features
Machine learning models cannot process words directly; text must be converted into numerical form. Two methods were implemented:
Method 1: CountVectorizer
	Represents text as a frequency matrix-each row= one review, each column= one unique word
	Values indicate how many times each word appears in that review
	Configured to include both single words and word pairs (ngram_range= (1,2))
	Limited to the top 5,000 most frequent words to balance detail with computational efficiency
	Simple and fast, but treats all words as equally important
Method 2: TF-IDF Vectorizer (Selected as Primary)
	Stands for Term Frequency-Inverse Document Frequency
	Measures how important a word is to a specific review, rather than just counting it
	Words that appear in nearly every review (e.g., “movie”, “film”) receive lower scores
	Rare distinctive words that strongly indicate sentiment (e.g., “masterpiece”, “disappointing”) receive higher scores
	Generally produces better results for sentiment analysis because it highlights meaningful, distinctive terms
	Same vocabulary limit of 5,000 features and inclusion of word pairs was applied
##Output Files
The following files were generated and saved:
	cleaned_reviews.csv- Original reviews alongside fully cleaned text
	tfidf_features.csv- Final TF-IDF feature matrix for model training
	count_features.csv- Alternative CountVectorizer feature matrix
-All source code and output files are committed to the shared GITHUB repository.
##Natural Language Processing for Sentiment Analysis
###Text Preprocessing
Natural language is unstructured and inconsistent. Standardization is essential because good computers treat “Good”, “good”, and “GOOD” as completely different tokens. Removing stop words reduces feature dimensionality and directs the model’s attention toward terms that actually convey sentiment.
##Bag-of-Words and CountVectorizer
The “bag-of-Words” model represent text as an unordered collection of words. CountVectorizer builds a vocabulary from the dataset and creates a sparse matrix where each entry is the raw frequency of a term in a document. Limitation: frequent words that are common across all documents dominate the representation.
##TF-IDF Vectorizer
TF-IDF addresses the limitation above by weighting term:
	Term Frequency (TF): How often a word appears in a specific review
	Inverse Document Frequency (IDF): How rare the word is across all reviews
	Final Score =TF × IDF: Rare, meaningful words get amplified; common words get suppressed
-This weighting scheme emphasizes the distinctive vocabulary that distinguishes positive reviews from negative ones, making TF-IDF well-suited to sentiment classification tasks.
