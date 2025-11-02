# Part 2: Indexing and Evaluation  
### IMPORTANT NOTE: This code is designed to run in Google Colab
## PROJECT OVERVIEW
This section implements the indexing and evaluation components for the fashion product retrieval system. It builds upon the preprocessed data from Part 1 to create efficient search capabilities and evaluate retrieval performance.
## SETUP INSTRUCTIONS FOR GOOGLE COLAB
### Set Your File Path - IMPORTANT!

```
  # Teacher Path - UPDATE WITH YOUR PATH
  docs_path =
```
Make sure your JSON file is in Google Drive and UPDATE THIS PATH with your actual file location of fashion_products_dataset.json.

You have to do the same in section 2.2.1, where you have to introduce your path to validation_labels.csv.

Finally, you have to download the ground_thurth.csv that contains the relevance for the top 10 documents of our queries (you can find it in this folder). Then, make sure your ground_thurth.csv is in Google Drive and set up the path in section 2.2.3.

Section 2 depends on the preprocessed data from Part 1. Ensure all Part 1 cells have been executed successfully before proceeding with Part 2.

After completing Part 1, run the Part 2 cells sequentially to:

- Create the inverted index from cleaned text
- Compute TF-IDF weights
- Rank queries
- Evaluate System Performance
- Analyze retrieval effectiveness

In the last cell of section 2.1.2 you can enter one of the proposed queries or try your own custom queries, to see how the system retrieves and ranks results.


### REMEMBER: The most important step is correctly setting both the docs_path, validation_path and ground_thurth.csv variables to point to your documents files in Google Drive. 
