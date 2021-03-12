# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:09:01 2021

@author: rouna
"""

import threading
D = {1:2,3:4,5:6}
def update(d):
    return {k:d[k]+1 for k in d}

def process(items, start, end):
    flag = 0
    for item in items[start:end]:                                               
        try:
            if flag == 0:
                d = update(D)
                flag += 1
            else:
                d = update(d)
            print("Item:",item+1,"dictionary:",d)
        except Exception:
            print('error with item')


def split_processing(items, num_splits=4):                                      
    split_size = len(items) // num_splits                                       
    threads = []
    for i in range(num_splits):
        # determine the indices of the list this thread will handle
        start = i * split_size
        # special case on the last chunk to account for uneven splits
        end = None if i+1 == num_splits else (i+1) * split_size
        # create the thread
        threads.append(                                                         
            threading.Thread(target=process, args=(items, start, end)))
        threads[-1].start() # start the thread we just created

    # wait for all threads to finish                                            
    for t in threads:                                                           
        t.join()                                                                


items = range(20)
split_processing(items)