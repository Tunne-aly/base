## Plot2Vec instructions

To plot the TSNE analysis of 100 most common words of your sentence data:

Run the plotter with `python plot2vec.py path/to/csv/data/folder 100`

If the number of words is not given 150 words are plotted.

The plotter uses the `nn` module's (formerly known as `sa`) get_sentences function to read the sentences from a csv file. It will build a WOrd2Vec model of the words, and run TSNE on the vectors in order to plot the most common words. Make sure you have your csv data in a folder, and that the sentences are in the second column. The plotter will print out all the plotted words as well as some similarity analysis. It also filters out some neutral and common words like `ja` and `mutta`.
