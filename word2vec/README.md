## Plot2Vec instructions

To plot the 100 most common words with a perplexity of 50 with TSNE of your sentence data:

Run the plotter with `python plot2vec.py path/to/csv/data/folder 100 50`

If amount of words and perplexity are not given, 150 words are plotted with a perplexity of 50.

The plotter will build a WOrd2Vec model of the words, and run TSNE on the vectors in order to plot the most common words. Make sure you have your csv data in a folder, and that the sentences are in the second column. The plotter will print out all the plotted words as well as some similarity analysis. It also filters out some neutral and common words like `ja` and `kuin`.
