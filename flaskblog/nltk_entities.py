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
   # print("\n".join([x[0] + '->' + ', '.join(x[1:]) for x in my_list]))
    #result = ""
    #for i in range(3):
        #result += f"{i[0]} -> {.join(i[1:])}\n"
    return "\n\n".join([x[0] + '->' + ', '.join(x[1:]) for x in my_list])

output = nltk_extraction(page)
print(output)

#print("\n".join([x[0] + '->' + ', '.join(x[1:]) for x in output]))
#    print(i[0], '->', ', '.join(i[1:]))

#for i in range(3):
#    print(i, '->',i+1)


#x = "#".join(myTuple)
#print(x)
#my_list1 = ["John", "Peter","Joe"]
#my_list2= ["x", "y"]

#result = ""
#for i in range(3):
 #   result += f"{my_list[i]} -> {.join(i[1:])}\n"

#print(result)
