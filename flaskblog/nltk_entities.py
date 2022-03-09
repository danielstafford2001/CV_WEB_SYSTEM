import nltk
from collections import defaultdict
 
page = """
EDUCATION	
University
Won first prize for the best second year group project, focused on software engineering.
Sixth Form
Mathematics, Economics, French
UK, London
"""

def nltk_extraction(text):
    entities = defaultdict(list)
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                entities[chunk.label()].append(''.join(c[0] for c in chunk))
    
    my_list = []
    for key, value in entities.items():
        list1= [key]
        my_list.append(list1 + value)
    
    return my_list

result = nltk_extraction(page)
print(result)

#for i in result:
 #   print(i[0], '->', ', '.join(i[1:]))