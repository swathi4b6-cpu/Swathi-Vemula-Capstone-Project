import os
import sqlite3
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import numpy as np

# --- CONSTANTS & CONFIGURATION ---
BASE_URL = "https://books.toscrape.com"
START_URL = "https://books.toscrape.com/catalogue/page-1.html"
DB_NAME = "catalog_benchmark.db"
FIXED_CONVERSION_RATE = 105.50  # Project-defined baseline constant (1 GBP = 105.50 INR)

RATING_MAP = {
    "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5
}

def extract_raw_data(max_pages=5):
    """Scrapes raw un-sanitized product data across paginated catalog lists."""
    print(f"[*] Initializing extraction from {BASE_URL}...")
    scraped_items = []
    current_url = START_URL
    pages_processed = 0

    while current_url and pages_processed < max_pages:
        try:
            response = requests.get(current_url, timeout=10)
            if response.status_code != 200:
                print(f"[!] Warning: Received status code {response.status_code} for {current_url}")
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            products = soup.find_all('article', class_='product_pod')
            
            for prod in products:
                title = prod.h3.a['title']
                price_text = prod.find('p', class_='price_color').text
                rating_class = prod.find('p', class_='star-rating')['class'][1]
                availability_text = prod.find('p', class_='instock availability').text.strip()

                product_href = prod.h3.a['href']
                product_url = urljoin(current_url, product_href)
                category_raw = "All Products Catalog"

                try:
                    detail_resp = requests.get(product_url, timeout=10)
                    if detail_resp.status_code == 200:
                        detail_soup = BeautifulSoup(detail_resp.text, 'html.parser')
                        breadcrumbs = detail_soup.select('ul.breadcrumb li')
                        if len(breadcrumbs) >= 3:
                            category_raw = breadcrumbs[-2].text.strip()
                except requests.RequestException:
                    pass

                scraped_items.append({
                    "title": title,
                    "price_raw": price_text,
                    "rating_raw": rating_class,
                    "availability_raw": availability_text,
                    "category_raw": category_raw
                })
            
            pages_processed += 1
            next_button = soup.find('li', class_='next')
            if next_button:
                current_url = urljoin(current_url, next_button.a['href'])
            else:
                current_url = None
                
        except requests.RequestException as e:
            print(f"[X] Network layer failure: {e}")
            break

    print(f"[+] Extraction complete. Extracted {len(scraped_items)} records across {pages_processed} pages.")
    return scraped_items

def transform_and_clean(raw_records):
    """Cleans raw string formats into strongly-typed elements with fallback sanitization."""
    print("[*] Initiating data transformations and currency enrichments...")
    df = pd.DataFrame(raw_records)
    
    # 1. Parse Price & Convert to numeric float
    # Removes currency symbols (e.g. £) via regex extraction
    df['price_gbp'] = df['price_raw'].apply(lambda x: re.sub(r'[^\d.]', '', str(x)))
    df['price_gbp'] = pd.to_numeric(df['price_gbp'], errors='coerce')
    
    # Missing Value Mitigation Strategy: Median Imputation
    # Handled systematically so unexpected formatting anomalies do not crash downstream execution
    if df['price_gbp'].isnull().any():
        median_price = df['price_gbp'].median()
        df['price_gbp'] = df['price_gbp'].fillna(median_price)
        print(f"[!] Imputed missing prices using catalog median value: £{median_price}")

    # 2. Convert Star Ratings String to Numerical Scale
    df['rating'] = df['rating_raw'].map(RATING_MAP)
    # If any structural parsing fails, default to a neutral median rating of 3
    df['rating'] = df['rating'].fillna(3).astype(int)

    # 3. Parse Availability Strings into Explicit Booleans
    df['in_stock'] = df['availability_raw'].apply(lambda x: 1 if "in stock" in str(x).lower() else 0)

    # 4. Currency Enrichment Calculation using Fixed Project Baseline
    df['price_inr'] = (df['price_gbp'] * FIXED_CONVERSION_RATE).round(2)
    
    # Clean up column selections
    df['category_name'] = df['category_raw'].str.strip().str.title()
    return df[['title', 'price_gbp', 'price_inr', 'rating', 'in_stock', 'category_name']]

def load_to_sqlite(df, database_path):
    """Initializes a normalized relational star layout and seeds records."""
    print(f"[*] Instantiating normalized relational DB engine at {database_path}...")
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Enforce foreign key constraints inside SQLite session
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Drop tables to ensure clean re-runs
    cursor.execute("DROP TABLE IF EXISTS books;")
    cursor.execute("DROP TABLE IF EXISTS categories;")

    # Initialize fully normalized multi-table database architecture
    cursor.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER,
        in_stock INTEGER,
        category_id INTEGER NOT NULL REFERENCES categories(category_id)
    );
    """)
    conn.commit()

    # Normalize Category Dim Table
    unique_categories = df['category_name'].unique()
    for cat in unique_categories:
        cursor.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?);", (cat,))
    conn.commit()

    # Map relational IDs back to primary records
    categories_df = pd.read_sql("SELECT * FROM categories", conn)
    category_map = dict(zip(categories_df['category_name'], categories_df['category_id']))
    df['category_id'] = df['category_name'].map(category_map)

    # Append mapped records directly into DB
    final_books_payload = df[['title', 'price_gbp', 'price_inr', 'rating', 'in_stock', 'category_id']]
    final_books_payload.to_sql('books', conn, if_exists='append', index=False)
    
    print("[+] Database loaded and normalized successfully.")
    conn.close()

if __name__ == "__main__":
    raw_data = extract_raw_data(max_pages=5)
    cleaned_df = transform_and_clean(raw_data)
    load_to_sqlite(cleaned_df, DB_NAME)
