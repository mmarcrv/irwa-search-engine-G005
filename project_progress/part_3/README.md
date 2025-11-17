# Part 3: Ranking & Filtering 
### IMPORTANT NOTE: This code is designed to run in Google Colab.
## SETUP INSTRUCTIONS FOR GOOGLE COLAB
### 1. Mount Google Drive
Before running any part of the notebook, make sure to mount Google Drive. This will allow the code to access the dataset stored in your Drive.
### 2. Set the file path (VERY IMPORTANT)
You must update the variable:
```
# Teacher Path - UPDATE WITH YOUR PATH
docs_path =
```
Make sure your json file is in Google Drive and UPDATE THIS PATH with your actual file location of fashion_products_dataset.json.
### 3. Install required libraries
The notebook automatically installs the external packages needed:

```
rank-bm25 (for BM25 ranking)
gensim (for Word2Vec)
```

The installation commands are already included in the first cells, so you only need to run them.
### 4. Running the notebook

Run all cells in order, top to bottom.

#### REMEMBER: The most important step is correctly setting the docs_path variable to point to your JSON file in Google Drive. 


