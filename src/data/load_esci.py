"""
Load and prepare Amazon ESCI dataset: merge examples + products, graded relevance, product text expansion.
"""
from __future__ import annotations  # Enable postponed evaluation of type hints (Python 3.7+)

from pathlib import Path  # For path manipulation

import pandas as pd  # For DataFrame operations

# ESCI label -> relevance score (higher = more relevant)
# Maps ESCI labels to numeric relevance scores for ranking
esci_label2relevance_pos = {
    "E": 4,  # Exact match - highest relevance
    "S": 2,  # Substitute - moderate relevance
    "C": 3,  # Complement - high relevance (often bought together)
    "I": 1,  # Irrelevant - lowest relevance
}

# Default paths: data lives in the clone at data/esci-data/shopping_queries_dataset/
REPO_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels from src/data/load_esci.py to project root
DATA_DIR = REPO_ROOT / "data"  # Path to data directory
ESCI_SUBDIR = DATA_DIR / "esci-data" / "shopping_queries_dataset"  # Path to ESCI parquet files

# Product expansion: max chars per field to stay within model max length
MAX_TITLE_LEN = 200  # Maximum characters for product title
MAX_DESC_LEN = 256  # Maximum characters for product description
MAX_BULLET_LEN = 300  # Maximum characters for bullet points


def _safe_str(x, max_len: int | None = None) -> str:
    """
    Safely convert value to string, handling None/NaN and truncating if needed.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):  # Check if value is None or NaN
        return ""  # Return empty string for missing values
    s = str(x).strip()  # Convert to string and remove leading/trailing whitespace
    if max_len is not None and len(s) > max_len:  # If truncation needed
        # Truncate at word boundary if possible, otherwise hard cut
        s = s[:max_len].rsplit(" ", 1)[0] if " " in s[:max_len] else s[:max_len]
    return s


def get_product_expanded_text(row: pd.Series) -> str:
    """
    Instacart-style product text expansion with special tokens.
    Format: [PN] title [PBN] brand [PBP] bullet [PD] description [PCL] color
    Creates a structured text representation for the product.
    """
    parts = []  # List to collect formatted parts
    # Extract and format title with [PN] prefix
    title = _safe_str(row.get("product_title"), MAX_TITLE_LEN)  # Get title, truncate if needed
    if title:  # Only add if non-empty
        parts.append(f"[PN] {title}")  # Add Product Name token + title
    # Extract and format brand with [PBN] prefix
    brand = _safe_str(row.get("product_brand"))  # Get brand (no truncation)
    if brand:  # Only add if non-empty
        parts.append(f"[PBN] {brand}")  # Add Product Brand Name token + brand
    # Extract and format bullet points with [PBP] prefix
    bullet = _safe_str(row.get("product_bullet_point"), MAX_BULLET_LEN)  # Get bullet points, truncate
    if bullet:  # Only add if non-empty
        parts.append(f"[PBP] {bullet}")  # Add Product Bullet Point token + bullets
    # Extract and format description with [PD] prefix
    desc = _safe_str(row.get("product_description"), MAX_DESC_LEN)  # Get description, truncate
    if desc:  # Only add if non-empty
        parts.append(f"[PD] {desc}")  # Add Product Description token + description
    # Extract and format color with [PCL] prefix
    color = _safe_str(row.get("product_color"))  # Get color (no truncation)
    if color:  # Only add if non-empty
        parts.append(f"[PCL] {color}")  # Add Product Color token + color
    # Join all parts with spaces, or return placeholder if empty
    return " ".join(parts) if parts else "[PN] (no title)"


def load_esci(
    data_dir: Path | str | None = None,
    *,
    small_version: bool = True,
    locale: str = "us",
    relevance_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Load ESCI examples + products, merge, filter, and add graded relevance (1-4).
    relevance_map: ESCI label -> relevance score (default esci_label2relevance_pos: E=4, S=2, C=3, I=1).
    small_version=True uses Task 1 reduced set; locale filters e.g. 'us'.
    """
    data_dir = Path(data_dir or ESCI_SUBDIR)  # Use provided path or default
    examples_path = data_dir / "shopping_queries_dataset_examples.parquet"  # Path to examples parquet
    products_path = data_dir / "shopping_queries_dataset_products.parquet"  # Path to products parquet
    if not examples_path.exists():  # Check if examples file exists
        raise FileNotFoundError(f"Examples not found: {examples_path}. Run download_esci first.")
    if not products_path.exists():  # Check if products file exists
        raise FileNotFoundError(f"Products not found: {products_path}.")

    df_ex = pd.read_parquet(examples_path)  # Load examples DataFrame (query-product pairs with labels)
    df_pr = pd.read_parquet(products_path)  # Load products DataFrame (product metadata)
    # Merge examples with product metadata on product_id and locale
    df = pd.merge(
        df_ex,  # Left DataFrame (examples)
        df_pr,  # Right DataFrame (products)
        on=["product_id", "product_locale"],  # Join keys
        how="left",  # Left join: keep all examples, add product info where available
    )

    if small_version:  # Filter to Task 1 reduced dataset (harder queries only)
        df = df[df["small_version"] == 1].copy()  # Keep only rows where small_version == 1
    df = df[df["product_locale"] == locale].copy()  # Filter by locale (e.g., "us", "es", "jp")

    mapping = relevance_map or esci_label2relevance_pos  # Use provided map or default
    # Map ESCI labels (E, S, C, I) to numeric relevance scores (4, 2, 3, 1)
    df["relevance"] = df["esci_label"].map(mapping).astype("int32")  # Create relevance column
    df = df.dropna(subset=["relevance"])  # Remove rows with unmapped labels (shouldn't happen with default map)

    # Expand product text with special tokens for each row
    df["product_text"] = df.apply(get_product_expanded_text, axis=1)  # Apply expansion row-wise
    return df  # Return merged DataFrame with relevance and expanded product text


def prepare_train_test(
    df: pd.DataFrame | None = None,
    data_dir: Path | str | None = None,
    **load_kw,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) with query, product_text, relevance (1-4), split from ESCI."""
    if df is None:  # If DataFrame not provided, load it
        df = load_esci(data_dir=data_dir or ESCI_SUBDIR, **load_kw)  # Load with provided kwargs
    train = df[df["split"] == "train"].copy()  # Filter rows where split == "train"
    test = df[df["split"] == "test"].copy()  # Filter rows where split == "test"
    return train, test  # Return train and test DataFrames
