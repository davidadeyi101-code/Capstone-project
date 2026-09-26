import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
print("Starting...\n")
#Load dataset
df=pd.read_csv("dataset/IMDB Dataset.csv")
print("Dataset loaded!")
print(f"Total reviews: {len(df)}")
print(f"columns found:{df.columns.tolist()}")
print("\n---First 2 reviews BEFORE cleaning---")
print(df['review'].head(2))
print()
#simple stopword list (no downloaded needed!)
SIMPLE_STOPWORDS = {
    'the', 'and', 'a', 'an', 'in', 'is', 'this', 'that', 'it', 'to', 'of', 'was', 'I', 'down', 'not', 'nor', 'too', 'never', 'ever',
    'on', 'at', 'her', 'his', 'them', 'they', 'she', 'he', 'you', 'we', 'me', 'my', 'only', 'even', 'into', 'out', 'up', 'still', 'while',
    'by', 'with', "for", 'as', 'but', 'or', 'so', 'if', 'than', 'then', 'more', 'most', 'much', 'many', 'such', 'also', 'after', 'before',
    'does', 'do', 'had', 'has', 'have', 'been', "were", 'are', 'be', 'what', 'who', 'which', 'their', 'its', 'just' ,
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'here', 'there', 'where', 'when', 'very', 'some',
    # Add more stopwords as needed
}
def clean_text(raw_text):
    text = str(raw_text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', ' ', text)  # keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    # Split into words and remove stopwords
    words = text.split()
    words = [w for w in words if w not in SIMPLE_STOPWORDS and len(w) > 2]
    return ' '.join(words)
#Clean all reviews
df['cleaned_review'] = df['review'].apply(clean_text)
print("Text cleaned!")
print("\n---Same reviews AFTER cleaning---")
print(df['cleaned_review'].head(2))
print()
#
print("=" * 50)
print("METHOD 1: CountVectorizer")
print("=" * 50)
count_vec = CountVectorizer(max_features=5000, ngram_range=(1,2))
X_count= count_vec.fit_transform(df['cleaned_review'])
print(f"Shape:{X_count.shape}")
#
print("\n" + "=" *50)
print("METHOD 2: TF-IDF Vectorizer")
print("=" *50)
tfidf= TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
X_tfidf= tfidf.fit_transform(df['cleaned_review'])
print(f"Shape:{X_tfidf.shape}")
#
df[['review', 'cleaned_review', 'sentiment']].to_csv("cleaned_reviews.csv", index=False)
tfidf_features = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())
tfidf_features.to_csv("tfidf_features.csv", index= False)
count_features = pd.DataFrame(X_count.toarray(), columns=count_vec.get_feature_names_out())
count_features.to_csv("count_features.csv", index=False)
print("\n ALL DONE! Files saved to your folder:")
print("cleaned_reviews.csv")
print("tfidf_features.csv")
print("Count_feature.csv")