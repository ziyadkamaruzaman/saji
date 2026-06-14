import os
from ..common import load_cleaned_reviews

def load_google_reviews(data_dir: str):
    """Load Google reviews"""
    path = os.path.join(data_dir, "GoogleReview_data_cleaned.csv")
    return load_cleaned_reviews(path, "google")