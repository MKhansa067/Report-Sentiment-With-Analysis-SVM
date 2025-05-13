#%% Library
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

#%% Set up visualization
sns.set_context('notebook')
sns.set_style('white')
plt.style.use('ggplot')

#%% Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def main():
    #%% Load data
    try:
        df = pd.read_csv("Training.txt", sep="\t", names=['liked', 'text'], encoding="utf-8")
        print("Data loaded successfully.")
        print(f"Shape: {df.shape}")
        print("\nFirst 3 rows:")
        print(df.head(3))
        print(f"\nTotal samples: {len(df)}")
        
        print("\nClass distribution:")
        print(df.groupby('liked').describe())
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    #%% Text processing functions
    def tokens(review):
        try:
            return TextBlob(str(review)).words
        except Exception as e:
            print(f"Error in tokenization: {e}")
            return []

    def split_into_lemmas(review):
        try:
            review = str(review).lower()
            words = TextBlob(review).words
            return [word.lemma for word in words]
        except Exception as e:
            print(f"Error in lemmatization: {e}")
            return []

    #%% Example text processing
    print("\nExample tokenization:")
    print(df.text.head().apply(tokens))
    
    print("\nExample lemmatization:")
    print(df.text.head().apply(split_into_lemmas))

    #%% Feature engineering
    try:
        print("\nCreating Bag-of-Words transformer...")
        bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(df['text'])
        print(f"Vocabulary size: {len(bow_transformer.vocabulary_)}")
        
        #%% Example transformation
        review1 = df['text'][3]
        print(f"\nExample review: {review1}")
        bow1 = bow_transformer.transform([review1])
        print(f"BoW representation shape: {bow1.shape}")
        print(f"Number of non-zero elements: {bow1.nnz}")
        
        #%% Transform entire dataset
        review_bow = bow_transformer.transform(df['text'])
        print('\nSparse matrix shape:', review_bow.shape)
        print('Number of non-zeros:', review_bow.nnz)
        print('Sparsity: %.2f%%' % (100.0 * review_bow.nnz / (review_bow.shape[0] * review_bow.shape[1])))
        
        #%% TF-IDF transformation
        tfidf_transformer = TfidfTransformer().fit(review_bow)
        review_tfidf = tfidf_transformer.transform(review_bow)
        print('\nTF-IDF matrix shape:', review_tfidf.shape)
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return

    #%% Model training
    try:
        #%% Split data
        text_train, text_test, liked_train, liked_test = train_test_split(
            df['text'], df['liked'], test_size=0.2, random_state=42
        )
        print(f"\nTrain size: {len(text_train)}, Test size: {len(text_test)}")
        
        #%% Define pipeline
        pipeline_svm = Pipeline([
            ('bow', CountVectorizer(analyzer=split_into_lemmas)),
            ('tfidf', TfidfTransformer()),
            ('classifier', SVC(probability=True)),
        ])

        #%% Hyperparameter grid
        param_svm = [
            {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
            {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
        ]

        #%% Grid search
        grid_svm = GridSearchCV(
            pipeline_svm,
            param_grid=param_svm,
            refit=True,
            n_jobs=-1,
            scoring='accuracy',
            cv=StratifiedKFold(n_splits=5),
            verbose=1
        )

        #%% Train model
        print("\nStarting model training...")
        start_time = time.time()
        classifier = grid_svm.fit(text_train, liked_train)
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds")
        
        #%% Results
        print("\nBest parameters:")
        print(classifier.best_params_)
        print(f"Best validation score: {classifier.best_score_:.4f}")
        
        print("\nClassification report on test set:")
        print(classification_report(liked_test, classifier.predict(text_test)))
        
        print("\nExample predictions:")
        print("'the vinci code is awesome':", classifier.predict(["the vinci code is awesome"])[0])
        print("'the vinci code is bad':", classifier.predict(["the vinci code is bad"])[0])
    except Exception as e:
        print(f"Error in model training: {e}")
        return

    #%% Gaussian kernel example
    def gaussKernel(x1, x2, sigma):
        ss = np.power(sigma, 2)
        norm = (x1 - x2).T.dot(x1 - x2)
        return np.exp(-norm/(2*ss))
    
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    print("\nGaussian kernel example:")
    print(f"Result: {gaussKernel(x1, x2, sigma)}")

if __name__ == "__main__":
    main()