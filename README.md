# Query arguments

## Information
This repository allows you to query the arguments from the 'arguments.json' by defining queries in the 'query.py' code. Different methods are used for matching the query with the arguments. Before running the code:

1. Install the required packages: ```pip install -r requirements.txt```
2. Define the queries in the ```queries``` list.
3. For the Jaccard Method and the Cosinus Similarity define the lower boundary where an argument is seen as relevant. For RANK_BM_25 and TF-IDF define how many arguments of the n most relevant arguments should be displayed.
4. Run the code: ```python query.py```
