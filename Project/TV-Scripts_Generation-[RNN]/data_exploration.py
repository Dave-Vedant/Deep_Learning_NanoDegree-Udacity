import numpy as np
import helper

view_line_range(0,10)

from collections import Counter

data_dir = './Data/Seinfeld_Scripts.txt'                        # relative path
text = helper.load_data(data_dir)

print("Dataset State")
print("Number of unique words: {}".format(len({word: None for word in text.split()})))

lines = text.split("\n")
print("No of lines : {}".format(len(lines)))

word_count_line = [len(line.split() for line in lines)]
print("Average Number of words per line : {}".format(np.average(word_count_line)))

print("\n The line {} to {}: ".format(*view_line_range))
print("\n".join(text.split("\n")[view_line_range[1]]))

