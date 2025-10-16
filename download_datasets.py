"""This script downloads the datasets used in the benchmark.

It currently downloads the Adult Income and Abalone datasets from the UCI
Machine Learning Repository.
"""
import os
import subprocess

def download_file(url, out_path):
    """Downloads a file from a URL to a specified path.

    This function checks if the file already exists at the output path and
    skips the download if it does.

    Args:
        url (str): The URL of the file to download.
        out_path (str): The local path where the file should be saved.
    """
    if not os.path.exists(out_path):
        print(f"Downloading {url} to {out_path}...")
        subprocess.run(["wget", "-O", out_path, url])
        print("Download complete.")
    else:
        print(f"{out_path} already exists. Skipping download.")


if __name__ == '__main__':
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Download Adult Income dataset
    adult_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    adult_path = "data/adult.csv"
    download_file(adult_url, adult_path)

    # Download Abalone dataset
    abalone_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    abalone_path = "data/abalone.csv"
    download_file(abalone_url, abalone_path)
