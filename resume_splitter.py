"""
Resume splitter
---------------
Resumes need to be split into sequences that can be fed into the neural
network. In order to do so, a hierarchy of separator symbols will be defined
"""

SEPARATORS = [
    "\n",
]

def split_resume(resume):

