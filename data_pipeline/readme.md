markdown# Zepto Catalog Competitor Intelligence & Benchmarking Pipeline

An automated production architecture built to ingest, sanitize, normalize, and evaluate unstructured public catalog arrays into standard schemas without dependencies on upstream live APIs.

## Pipeline Architecture & Engineering Design Decisions

### 1. Data Ingestion & Fault-Tolerant Extractor
- Implements modular web scraping with `BeautifulSoup` pointing to `books.toscrape.com`.
- Safely processes paginated structures up to 5 complete catalog sequences, generating a minimum baseline payload dataset of 100 deep records (surpassing structural requirement parameters of >= 60 entries).

### 2. Defensively Encapsulated Data Transformations
- **Price Cleaning**: Automatically removes character strings (`£`) and forces numeric conversions using regular expressions.
- **Robust Anomaly Mitigation**: If structural layout anomalies inject invalid fields or strings, data mutations automatically handle exceptions via **Median Imputation** for continuous measurements (e.g., `price_gbp`), or uniform default fallbacks for categorical variables. This prevents execution runtime crashes during bad data cycles.
- **Currency Mapping Engine**: Enhances fields via an immutable project-wide baseline fixed constant computation:
  $$\text{Price (INR)} = \text{Price (GBP)} \times 105.50$$
  This removes network latency or key dependency updates during automated continuous integration cycles.

### 3. Star Schema Relational Blueprint
Data is split into structural normalized tables inside an ACID-compliant `SQLite` relational layer to limit storage footprint overheads:
- `categories`: Dimensions catalog (`category_id` Autoincrement PK, Unique Name text index).
- `books`: Primary transaction registry records containing pricing variables, explicit rating validation constraints, and a strict constraint `FOREIGN KEY` link referencing `categories(category_id)`.
