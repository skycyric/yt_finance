import pandas as pd
import psycopg2
from psycopg2 import sql
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from transformers import logging
import requests
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import os
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Function to connect to the database


def connect_to_db():
    engine = create_engine(
        'postgresql://tvbs:10030805@localhost:5432/postgres')
    return engine

# Function to fetch table names from the database


def fetch_table_names(engine):
    query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'yt_data' AND table_name != 'classification_results';
    """
    return pd.read_sql(query, engine)['table_name'].tolist()

# Function to fetch video data from the database


def fetch_video_data(engine, table_name):
    query = f"SELECT * FROM yt_data.{table_name};"
    data = pd.read_sql(query, engine)
    # 調試用的打印語句
    print(f"Fetched columns for table {table_name}: {data.columns.tolist()}")
    # 確保 'brand' 欄位存在
    if 'brand' not in data.columns:
        data['brand'] = table_name  # 如果沒有 'brand' 欄位，使用表名作為品牌
    return data

# Function to classify text using BERT


def classify_text_bert(texts):
    logging.set_verbosity_error()  # Suppress warnings
    tokenizer = None
    model = None
    retries = 3
    for _ in range(retries):
        try:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased')
            break
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            print(f"Error downloading model: {e}. Retrying...")
    if tokenizer is None or model is None:
        raise Exception(
            "Failed to download BERT model after multiple attempts.")

    # Truncate or pad texts to the maximum sequence length (512 tokens)
    max_length = 512
    texts = [text[:max_length] for text in texts]

    classifier = pipeline('sentiment-analysis',
                          model=model, tokenizer=tokenizer)
    return classifier(texts)

# Function to classify text using LDA


def classify_text_lda(texts, num_topics=5):
    chinese_stop_words = [
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'
    ]
    vectorizer = CountVectorizer(stop_words=chinese_stop_words)
    text_matrix = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(text_matrix)
    topics = lda.transform(text_matrix)
    return topics

# Function to alter the classification_results table to add title_vector column if it doesn't exist


def alter_classification_results_table(engine):
    conn = psycopg2.connect(
        host='localhost',
        dbname='postgres',
        user='tvbs',
        password='10030805'
    )
    cur = conn.cursor()
    # Create table if not exists
    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS yt_data.classification_results (
            video_id TEXT PRIMARY KEY,
            brand TEXT,
            title TEXT,
            tags TEXT[],
            description TEXT,
            views INTEGER,
            upload_date DATE,
            lda_classification INTEGER,
            bert_classification TEXT,
            href TEXT,
            title_vector VECTOR
        );
    """))
    conn.commit()
    # Check if title_vector column exists and has the correct type
    cur.execute(sql.SQL("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'yt_data' 
        AND table_name = 'classification_results' 
        AND column_name = 'title_vector';
    """))
    column_info = cur.fetchone()
    if column_info is None or column_info[1] != 'vector':
        # Drop the column if it exists but has the wrong type
        cur.execute(sql.SQL("""
            ALTER TABLE yt_data.classification_results 
            DROP COLUMN IF EXISTS title_vector;
        """))
        conn.commit()
        # Add the column with the correct type
        cur.execute(sql.SQL("""
            ALTER TABLE yt_data.classification_results
            ADD COLUMN title_vector VECTOR;
        """))
        conn.commit()
    cur.close()
    conn.close()

# Function to save classification results to the database


def save_classification_results(engine, results):
    # Ensure the table has the title_vector column
    alter_classification_results_table(engine)
    conn = psycopg2.connect(
        host='localhost',
        dbname='postgres',
        user='tvbs',
        password='10030805'
    )
    cur = conn.cursor()

    # Create table if not exists
    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS yt_data.classification_results (
            video_id TEXT PRIMARY KEY,
            brand TEXT,
            title TEXT,
            tags TEXT[],
            description TEXT,
            views INTEGER,
            upload_date DATE,
            lda_classification INTEGER,
            bert_classification TEXT,
            href TEXT,
            title_vector VECTOR
        );
    """))
    conn.commit()

    # Insert or update classification results
    for result in results:
        # 打印每個結果以確保 href 列存在並且有值
        print(f"Inserting result: {result}")
        # Handle NaN values in title_vector
        result['title_vector'] = np.nan_to_num(result['title_vector'])
        # Ensure all data types are correct
        result['video_id'] = str(result['video_id'])
        result['brand'] = str(result['brand'])
        result['title'] = str(result['title'])
        result['tags'] = list(result['tags'])
        result['description'] = str(result['description'])
        result['views'] = int(result['views'])
        result['upload_date'] = pd.to_datetime(result['upload_date']).date()
        result['lda_classification'] = int(result['lda_classification'])
        result['bert_classification'] = str(result['bert_classification'])
        result['href'] = str(result['href'])
        # Convert numpy.ndarray to list for pgvector
        result['title_vector'] = result['title_vector'].tolist()

        cur.execute(sql.SQL("""
            INSERT INTO yt_data.classification_results (video_id, brand, title, tags, description, views, upload_date, lda_classification, bert_classification, href, title_vector)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (video_id) DO UPDATE SET
                brand = EXCLUDED.brand,
                title = EXCLUDED.title,
                tags = EXCLUDED.tags,
                description = EXCLUDED.description,
                views = EXCLUDED.views,
                upload_date = EXCLUDED.upload_date,
                lda_classification = EXCLUDED.lda_classification,
                bert_classification = EXCLUDED.bert_classification,
                href = EXCLUDED.href,
                title_vector = EXCLUDED.title_vector;
        """), (
            result['video_id'],
            result['brand'],
            result['title'],
            result['tags'],
            result['description'],
            result['views'],
            result['upload_date'],
            result['lda_classification'],
            result['bert_classification'],
            result['href'],
            result['title_vector']
        ))
    conn.commit()
    cur.close()
    conn.close()

# Function to save similar videos to the database


def save_similar_videos(engine, similar_videos):
    conn = psycopg2.connect(
        host='localhost',
        dbname='postgres',
        user='tvbs',
        password='10030805'
    )
    cur = conn.cursor()

    # Create table if not exists
    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS yt_data.similar_videos (
            main_video_id TEXT,
            main_video_title TEXT,
            main_video_url TEXT,
            main_video_views INTEGER,
            similar_video_id TEXT,
            similar_video_title TEXT,
            similar_video_url TEXT,
            similarity_score FLOAT,
            similar_video_brand TEXT
        );
    """))
    conn.commit()

    # Insert similar videos
    for _, row in similar_videos.iterrows():
        # Fetch the correct brand for similar_video_id
        cur.execute(sql.SQL("""
            SELECT brand FROM yt_data.classification_results WHERE video_id = %s;
        """), (row['similar_video_id'],))
        similar_video_brand = cur.fetchone()

        if similar_video_brand:
            similar_video_brand = similar_video_brand[0]
        else:
            similar_video_brand = 'Unknown'  # Handle case where brand is not found

        cur.execute(sql.SQL("""
            INSERT INTO yt_data.similar_videos (main_video_id, main_video_title, main_video_url, main_video_views, similar_video_id, similar_video_title, similar_video_url, similarity_score, similar_video_brand)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (main_video_id, similar_video_id) DO NOTHING;
        """), (
            row['main_video_id'],
            row['main_video_title'],
            row['main_video_url'],
            row['main_video_views'],
            row['similar_video_id'],
            row['similar_video_title'],
            row['similar_video_url'],
            row['similarity_score'],
            similar_video_brand
        ))
    conn.commit()
    cur.close()
    conn.close()

# Function to normalize vectors


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return vectors / norms

# Function to find similar videos to tvbs_money_playlist videos


def find_similar_videos(engine):
    query = "SELECT * FROM yt_data.classification_results;"
    data = pd.read_sql(query, engine)

    if data.empty:
        print("No data available for similarity search.")
        return pd.DataFrame()

    # Filter tvbs_money_playlist videos
    main_videos = data[data['brand'] ==
                       'tvbs_money_playlist'].reset_index(drop=True)
    other_videos = data[data['brand'] !=
                        'tvbs_money_playlist'].reset_index(drop=True)

    print(f"Main videos count: {len(main_videos)}")
    print(f"Other videos count: {len(other_videos)}")

    if main_videos.empty:
        print("No tvbs_money_playlist videos found.")
        return pd.DataFrame()

    if other_videos.empty:
        print("No other videos found.")
        return pd.DataFrame()

    # Convert title_vector from string to list of floats
    main_videos['title_vector'] = main_videos['title_vector'].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=','))
    other_videos['title_vector'] = other_videos['title_vector'].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=','))

    # Ensure all vectors have the same shape
    vector_length = main_videos['title_vector'].iloc[0].shape[0]
    main_videos['title_vector'] = main_videos['title_vector'].apply(
        lambda x: x if x.shape[0] == vector_length else np.zeros(vector_length))
    other_videos['title_vector'] = other_videos['title_vector'].apply(
        lambda x: x if x.shape[0] == vector_length else np.zeros(vector_length))

    # Normalize vectors
    main_title_vectors = np.stack(main_videos['title_vector'].values)
    other_title_vectors = np.stack(other_videos['title_vector'].values)
    main_title_vectors = normalize_vectors(main_title_vectors)
    other_title_vectors = normalize_vectors(other_title_vectors)

    # Handle NaN values in title vectors
    main_title_vectors = np.nan_to_num(main_title_vectors)
    other_title_vectors = np.nan_to_num(other_title_vectors)

    # Debug print statements to check vectors
    print(f"Main title vectors: {main_title_vectors}")
    print(f"Other title vectors: {other_title_vectors}")

    title_similarity = cosine_similarity(
        main_title_vectors, other_title_vectors)

    similar_videos = []
    similarity_threshold = 0.4  # Adjusted similarity threshold
    for idx, main_video in main_videos.iterrows():
        for similar_idx in range(len(other_videos)):
            similar_video = other_videos.iloc[similar_idx]
            similarity_score = title_similarity[idx][similar_idx]
            date_diff = abs(
                (main_video['upload_date'] - similar_video['upload_date']).days)
            # Debug print statements
            print(
                f"Main video ID: {main_video['video_id']}, Similar video ID: {similar_video['video_id']}")
            print(
                f"Similarity score: {similarity_score}, Date difference: {date_diff} days")
            if similarity_score > similarity_threshold and date_diff <= 30:
                similar_videos.append({
                    'main_video_id': main_video['video_id'],
                    'main_video_title': main_video['title'],
                    'main_video_url': main_video['href'],
                    'main_video_views': main_video['views'],
                    'similar_video_id': similar_video['video_id'],
                    'similar_video_title': similar_video['title'],
                    'similar_video_url': similar_video['href'],
                    'similarity_score': similarity_score,
                    'similar_video_brand': similar_video['brand']
                })

    print(f"Similar videos found: {len(similar_videos)}")
    return pd.DataFrame(similar_videos)

# Main function to classify video data and save results


def main():
    engine = connect_to_db()
    table_names = fetch_table_names(engine)

    all_results = []

    for table_name in table_names:
        data = fetch_video_data(engine, table_name)
        if data.empty:
            continue

        # 確保 'title' 列存在
        if 'title' not in data.columns:
            print(f"Column 'title' not found in table {table_name}")
            continue
        if 'brand' not in data.columns:
            data['brand'] = table_name  # 如果沒有 'brand' 欄位，使用表名作為品牌

        # 移除缺少 'title' 的行
        data = data.dropna(subset=['title'])

        # Classify titles, tags, and descriptions
        bert_classifications = classify_text_bert(data['title'].tolist())
        lda_classifications = classify_text_lda(data['title'].tolist())

        # Prepare results
        chinese_stop_words = [
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'
        ]
        vectorizer = CountVectorizer(stop_words=chinese_stop_words)
        title_vectors = vectorizer.fit_transform(
            data['title'].tolist()).toarray()

        # Reduce dimensionality of title_vectors
        # Ensure n_components is <= n_features
        n_components = min(200, title_vectors.shape[1])
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_title_vectors = svd.fit_transform(title_vectors)

        for i, row in data.iterrows():
            all_results.append({
                'video_id': row['video_id'],
                'brand': table_name,
                'title': row['title'],
                'tags': row['tags'],
                'description': row['description'],
                'views': row['views'],
                'upload_date': row['upload_date'],
                'lda_classification': lda_classifications[i].argmax(),
                'bert_classification': bert_classifications[i]['label'],
                'href': row['href'],
                'title_vector': reduced_title_vectors[i]
            })

    # Convert all_results to DataFrame
    results_df = pd.DataFrame(all_results)

    # 確保 'title' 列存在於結果中
    if 'title' not in results_df.columns:
        print("Column 'title' not found in results_df")
        return

    # Convert results_df back to list of dictionaries
    all_results = results_df.to_dict('records')

    # Save all results to the database
    save_classification_results(engine, all_results)

    # Find similar videos to tvbs_money_playlist videos
    similar_videos_df = find_similar_videos(engine)
    print(similar_videos_df)

    # Save similar videos to the database
    save_similar_videos(engine, similar_videos_df)


if __name__ == '__main__':
    main()
