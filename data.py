
# coding: utf-8

# In[64]:


#! /usr/bin/env python
import os
import argparse
import datetime
import re
import numpy as np
import random

#import spacy
#spacy_en = spacy.load('en')

from nltk.tokenize import word_tokenize


# In[65]:


def create_one_batch(ids, x, y):
    batch_x = np.row_stack( [ [x[i]] for i in ids ] )
    batch_y = np.array( [ y[i] for i in ids ] )
    return batch_x, batch_y


# shuffle training examples and create mini-batches
def create_batches(x, y, batch_size, shuffle = True):
    
    perm = list(range(len(y)))
    if shuffle:
        random.shuffle(perm)

    # sort sequences based on their length
    # permutation is necessary if we want different batches every epoch
    lst = sorted(perm, key=lambda i: len(x[i][0]))
    #print(lst)
    batches_x = [ ]
    batches_y = [ ]
    size = batch_size
    ids = [ lst[0] ]
    for i in lst[1:]:
        if len(ids) < size and len(x[i][0]) == len(x[ids[0]][0]):
            ids.append(i)
        else:
            bx, by = create_one_batch(ids, x, y)
            batches_x.append(bx)
            batches_y.append(by)
            ids = [ i ]
    bx, by = create_one_batch(ids, x, y)
    batches_x.append(bx)
    batches_y.append(by)

    # shuffle batches
    batch_perm = list(range(len(batches_x)))
    random.shuffle(batch_perm)
    batches_x = [ batches_x[i] for i in batch_perm ]
    batches_y = [ batches_y[i] for i in batch_perm ]
    return batches_x, batches_y


# In[66]:


def tokenizer(text):
    #return " ".join([tok.text for tok in spacy_en.tokenizer(text)])
    #print(text)
    #print(word_tokenize(text))
    return " ".join(word_tokenize(text))


# In[67]:


def data_split(datas, test_rate = 0.2):
    random.shuffle(datas)
    pivot_index = int(len(datas) * (1-test_rate))
    return datas[:pivot_index], datas[pivot_index:]


# In[68]:



p1 = '<e1>.+</e1>'
p2 = '<e2>.+</e2>'

def SemEval10_task8(sub_path, path = './dataset/SemEval2010_task8_all_data/'):
    full_path = path + sub_path
    
    fin = open(full_path, 'r')
    lines = fin.readlines()
    
    datas = []
    
    for i in range(0, len(lines), 4):
        data = {'content': [],
                 'category': []}
        
        content_line = lines[i].strip()
        if not content_line:
            break
        _, raw_sentence = content_line.split('\t')
        raw_sentence = raw_sentence[1:-1]
        
        #print(raw_sentence)
        #
        e1_span = re.search(p1, raw_sentence).span()
        e1 = raw_sentence[e1_span[0]+4:e1_span[1]-5]
        
        
        e2_span = re.search(p2, raw_sentence).span()
        e2 = raw_sentence[e2_span[0]+4:e2_span[1]-5]
        
        temp_sentence = re.sub(p1, ' E1 ', raw_sentence)
        temp_sentence = re.sub(p2, ' E2 ', temp_sentence)
        temp_sentence = re.sub('  ', ' ', temp_sentence)
        temp_sentence = tokenizer(temp_sentence)
        #print(temp_sentence)
        data['content'] = (temp_sentence, e1, e2)
        
        category_line = lines[i+1].strip()
        if category_line != 'Other':
            data['category'] = (category_line[:-7], category_line[-5], category_line[-2])
        else:
            data['category'] = (category_line,None,None)
        
        comment_line = lines[i+2].strip()
        
        datas.append(data)
    return datas
        
        
    
    


# In[69]:


"""
def preprocess_dataset(datas, MAX_POS = 15, label_factorize = False, entity_normalize = False):
    sents = []
    position_indices_1= []
    position_indices_2= []
    labels = []
    for data in datas:
        # Get content data from input data
        sentence, e1, e2 = data['content']
        
        # Change entities as a original word
        words = sentence.split()
        e1_index = e2_index = None
        for i, word in enumerate(words):
            if word == 'E1':
                if not entity_normalize:
                    words[i] = e1
                e1_index = i
            elif word == 'E2':
                if not entity_normalize:
                    words[i] = e2
                e2_index = i
        sents.append(words)
        #print(words)
        
        # Get position embedding index
        position_index_1 = []
        position_index_2 = []
        for i in range(len(words)):
            position_index_1.append(int(MAX_POS * abs(i - e1_index) / (i - e1_index)) if abs(i - e1_index) > MAX_POS else i - e1_index)
            position_index_2.append(int(MAX_POS * abs(i - e2_index) / (i - e2_index)) if abs(i - e2_index) > MAX_POS else i - e2_index)
            
            position_index_1[i] += MAX_POS
            position_index_2[i] += MAX_POS
        position_indices_1.append(position_index_1)
        position_indices_2.append(position_index_2)
        #print(position_index_1)
        #print(position_index_2)
        
        # Make label
        if not label_factorize:
            #print(data['category'])
            label, t1, t2 = data['category']
            if label != 'Other':
                labels.append(label + '(e' + t1+',e'+ t2+')')
            else:
                labels.append(label)
    
    return list(zip(sents, position_indices_1, position_indices_2)), labels
    #return {'sents': sents, 'position_indices_1': position_indices_1, 'position_indices_2': position_indices_2, 'labels': labels}
"""


# In[70]:


def preprocess_dataset(datas, MAX_POS = 15, label_factorize = False, entity_normalize = False, directional_consideration = True):
    sents = []
    position_indices_1= []
    position_indices_2= []
    labels = []
    for data in datas:
        # Get content data from input data
        sentence, e1, e2 = data['content']
        
        # Change entities as a original word
        words = sentence.split()
        e1_index = e2_index = None
        e1_len = e2_len = None
        input_words = []
        
        index = 0
        for word in words:
            #print(input_words)
            if word == 'E1':
                if entity_normalize:
                    input_words.append('e1')
                    index += 1
                else:
                    
                    tokens = e1.strip().split(' ')
                    
                    input_words += ['<e1>']
                    for token in tokens:
                        
                        input_words += [token]
                    input_words += ['</e1>']
                    e1_index = index
                    e1_len = len(tokens)+2
                    index += e1_len
            elif word == 'E2':
                if entity_normalize:
                    input_words.append('e2')
                    index += 1
                else:
                    
                    tokens = e2.strip().split(' ')
                    
                    input_words += ['<e2>']
                    for token in tokens:
                        input_words += [token]
                    input_words += ['</e2>']
                    e2_index = index
                    e2_len = len(tokens)+2
                    index += e2_len
            else:
                input_words += [word]
                index += 1
        sents.append(input_words)
        #print(input_words)
        #print(words)
        
        # Get position embedding index
        position_index_1 = []
        position_index_2 = []
        for i in range(len(input_words)):
            
            if i < e1_index:
                pos1 =   i - e1_index
            elif i >= e1_index + e1_len:
                pos1 = i - (e1_index + e1_len - 1)
            else:
                pos1 = 0
            
            
            if i < e2_index:
                pos2 =   i - e2_index
            elif i >= e2_index + e2_len:
                pos2 = i - (e2_index + e2_len - 1)
            else:
                pos2 = 0
            #print(pos2)
            position_index_1.append(int(MAX_POS * abs(pos1) / (pos1)) if abs(pos1) > MAX_POS else pos1)
            position_index_2.append(int(MAX_POS * abs(pos2) / (pos2)) if abs(pos2) > MAX_POS else pos2)
            #position_index_2.append(pos2 + MAX_POS)
            #position_index_1.append(pos1 + MAX_POS)
            #print(input_words[e1_index:e1_index+e1_len])
            #print(input_words[e2_index:e2_index+e2_len])
            
            
            position_index_1[i] += MAX_POS
            position_index_2[i] += MAX_POS
        position_indices_1.append(position_index_1)
        position_indices_2.append(position_index_2)
        #print(position_index_1)
        #print(position_index_2)
        
        # Make label
        if not label_factorize:
            #print(data['category'])
            label, t1, t2 = data['category']
            if label != 'Other':
                if directional_consideration:
                    labels.append(label + '(e' + t1+',e'+ t2+')')
                else:
                    labels.append(label)
            else:
                labels.append(label)
    
    return list(zip(sents, position_indices_1, position_indices_2)), labels


# In[71]:


#train = SemEval10_task8(sub_path='SemEval2010_task8_training/TRAIN_FILE.TXT')
#train_data = preprocess_dataset(train)


# In[72]:


#train_input = train_data[0]
#train_output = train_data[1]


# In[73]:


#train_input[0]


# In[74]:


#batch_x, batch_y = create_batches(train_input, train_output, 16)


# In[75]:


#print(len(batch_y[0]))
#print(len(batch_y))
#print(len(batch_x[0]))
#print(len(batch_x))
#print(batch_x[0])
#print(batch_y[0])


# In[76]:


#torch.Longtensor([1,2,3,4,5])

