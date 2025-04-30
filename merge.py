import pandas as pd

# Load your scraped articles
scraped_df = pd.read_csv('news_articles_labeled.csv')

# Load Kaggle dataset
kaggle_df = pd.read_csv('kaggle_dataset/Fake.csv')
true_df = pd.read_csv('kaggle_dataset/True.csv')

# Clean/standardize scraped data
scraped_df = scraped_df.rename(columns={
    'url': 'link',
    'content': 'description'
})

scraped_df['source'] = scraped_df.get('source', 'Scraped News')  # Ensure 'source' exists
scraped_df['label'] = scraped_df.get('label', 'unknown')  # Ensure 'label' exists, default to 'unknown'
scraped_df['scraped_at'] = scraped_df.get('scraped_at', pd.Timestamp.utcnow().isoformat())  # Ensure 'scraped_at' exists

# If 'published' is missing, fill with current UTC time
scraped_df['published'] = scraped_df['published'].fillna(pd.Timestamp.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT'))

# Add missing columns from Kaggle dataset to scraped_df (if any, like 'subject')
scraped_df['subject'] = scraped_df.get('subject', 'unknown')  # Kaggle has 'subject', so add it with default

# Ensure all Kaggle columns are present and standardized
kaggle_df = kaggle_df.rename(columns={'text': 'description', 'date': 'published'})
true_df = true_df.rename(columns={'text': 'description', 'date': 'published'})

# Add missing columns to Kaggle datasets to match scraped_df
for df in [kaggle_df, true_df]:
    df['source'] = df.get('source', 'Kaggle News')
    df['source_type'] = df.get('source_type', 'unknown')
    df['scraped_at'] = df.get('scraped_at', pd.Timestamp.utcnow().isoformat())
    df['fact_check_link'] = df.get('fact_check_link', '')
    df['fact_check_verdict'] = df.get('fact_check_verdict', '')

# Assign labels to Kaggle datasets
kaggle_df['label'] = 'fake'
true_df['label'] = 'true'

# Merge the dataframes
merged_df = pd.concat([scraped_df, kaggle_df, true_df], ignore_index=True)

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_news.csv', index=False)