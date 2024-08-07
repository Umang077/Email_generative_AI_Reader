import os
import re
import email
from email import policy
from email.parser import BytesParser
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
 
# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')
 
# Function to extract subject and body from .eml files
def extract_subject_and_body(eml_file):
    with open(eml_file, 'rb') as file:
        msg = BytesParser(policy=policy.default).parse(file)
   
    subject = msg['subject'] if msg['subject'] else ""
    body = ""
 
    if msg.is_multipart():
        for part in msg.iter_parts():
            if part.get_content_type() == 'text/plain':
                charset = part.get_content_charset()
                if charset:
                    body = part.get_payload(decode=True).decode(charset, errors='replace')
                else:
                    body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                break
    else:
        charset = msg.get_content_charset()
        if charset:
            body = msg.get_payload(decode=True).decode(charset, errors='replace')
        else:
            body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
   
    return str(subject), str(body)
 
# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)
 
# Path to the folder containing .eml files
eml_folder = r'C:\\Users\\shivamsinghrajput\\Music\\data'
 
# Path to the folder where cluster text files will be saved
cluster_output_folder = r'C:\\Users\\shivamsinghrajput\\Music\\cluster_output'
 
# Ensure the output directory exists
os.makedirs(cluster_output_folder, exist_ok=True)
 
# Read all .eml files and extract subjects and bodies
emails = []
 
for filename in os.listdir(eml_folder):
    if filename.endswith('.eml'):
        subject, body = extract_subject_and_body(os.path.join(eml_folder, filename))
        emails.append({"subject": subject, "body": body})
 
# Prepare data
X = [preprocess_text(email['subject'] + " " + email['body']) for email in emails]
 
# Vectorize the text data
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
X_vectorized = vectorizer.fit_transform(X)
 
# Determine the optimal number of clusters using the elbow method
def plot_elbow_curve(X):
    wcss = []
    for i in range(1, 21):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 21), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
 
plot_elbow_curve(X_vectorized)
 
# Choose the number of clusters
n_clusters = 15  # Updated number of clusters
 
# Perform K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_vectorized)
 
# Assign descriptive names to clusters (update with meaningful names after analysis)
cluster_names = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
    4: "Cluster 4",
    5: "Cluster 5",
    6: "Cluster 6",
    7: "Cluster 7",
    8: "Cluster 8",
    9: "Cluster 9",
    10: "Cluster 10",
    11: "Cluster 11",
    12: "Cluster 12",
    13: "Cluster 13",
    14: "Cluster 14"
}
 
# Add cluster labels to emails
for i, email in enumerate(emails):
    email['cluster'] = y_kmeans[i]
 
def clean_text(text):
    # Replace problematic characters with a placeholder or remove them
    return text.encode('ascii', 'ignore').decode('ascii')
 
# Write clustered emails to text files
for cluster_id, cluster_name in cluster_names.items():
    with open(os.path.join(cluster_output_folder, f'{cluster_name}.txt'), 'w', encoding='utf-8') as file:
        for email in emails:
            if email['cluster'] == cluster_id:
                subject_clean = clean_text(email['subject'])
                body_clean = clean_text(email['body'])
                file.write(f"Subject: {subject_clean}\n")
                file.write(f"Body:\n{body_clean}\n")
                file.write("="*40 + "\n\n")
 
# Visualize the clusters using PCA (optional)
def plot_clusters(X, y_kmeans):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis')
    plt.title('Email Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()
 
plot_clusters(X_vectorized, y_kmeans)
