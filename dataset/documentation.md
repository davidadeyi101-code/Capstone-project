#Text Preprocessing and Vectorization
##Project Overview
This project builds a Sentiment Analyzer that reads movie review and predicts if the sentiment is Positive or Negative. The workflow includes:
	Loading and cleaning movie review text data
	Preprocessing text to prepare it for modelling
	Converting text into numerical features using CountVectorizer
	Training and evaluating 3 ML models
	Selecting the best-performing model
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
Machine learning models cannot process words directly; text must be converted into numerical form. Methods implemented include:
Method 1: CountVectorizer
	Represents text as a frequency matrix-each row= one review, each column= one unique word
	Values indicate how many times each word appears in that review
	Configured to include both single words and word pairs (ngram_range= (1,2))
	Limited to the top 5,000 most frequent words to balance detail with computational efficiency
	Simple and fast, but treats all words as equally important
###Why CountVectorizer?
	It is faster to train and run (has less computation time)
	Simple to understand and works perfectly for this dataset
	Count number of times each word appears in reviews
-How it works:
	Builds vocabulary
	Turns every review into a row of numbers
	Each number = how many times that word appears

##Output Files
The following files were generated and saved:
	cleaned_reviews.csv- Original reviews alongside fully cleaned text
	count_features.csv- Alternative CountVectorizer feature matrix
All source code and output files are committed to the shared GITHUB repository.
##Natural Language Processing for Sentiment Analysis
-Text Preprocessing
Natural language is unstructured and inconsistent. Standardization is essential because good computers treat “Good”, “good”, and “GOOD” as completely different tokens. Removing stop words reduces feature dimensionality and directs the model’s attention toward terms that actually convey sentiment.
-Bag-of-Words and CountVectorizer
The “bag-of-Words” model represent text as an unordered collection of words. CountVectorizer builds a vocabulary from the dataset and creates a sparse matrix where each entry is the raw frequency of a term in a document. Limitation: frequent words that are common across all documents dominate the representation.
##Models and Evaluation
We trained 3 models and measured performance using:
	Accuracy: % of total predictions correct
	Precision: How many predicted positives were actually positive
	F1 score: Balanced measure combining precision and recall
	Prediction Performance: How well the model works on new, unseen reviews
###Model Results:
Model	Accuracy	Precision	F1 score	Prediction\ Performance
Logistic Regression	0.88	0.87	0.88	Best across all metrics, fastest, reliable on new data
Support Vector Machine (SVM)	0.86	0.85	085	Good, slightly slower
Random Forest	0.83	0.82	0.82	Fair, tends to overfit; lowest scores

Best Performing model Selected:
LOGISTIC REGRESSION
Reason:
	Highest accuracy, precision and F1 score
	Fastest training and prediction
	Most reliable on unseen reviews
	Best all-around performance
##Conclusion
	CountVectorizer successfully converted text efficiently and quickly
	Logistic Regression outperformed SVM and Random Forest though they both gave solid results
