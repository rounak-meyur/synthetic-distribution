# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:46:27 2019

Author: Rounak Meyur
"""

import os
import matplotlib.pyplot as plt
import imageio

class ImageHelper(object):
    """
    """
    def save_image(self,figure,name,directory,close=True):
        '''
        '''
        figure.savefig("{}{}.png".format(directory,name))
        if close: plt.close()
        return
    
    
    def makegif(self,src,dest):
        '''
        Input:  src : Source directory of images
                dest: Destination path of gif
        '''
        fnames = [f for f in os.listdir(src) if ".png" in f]
        fnames_sorted = [str(m)+'.png'for m in 
                         sorted([int(s.strip('.png')) for s in fnames])]
        
    
        with imageio.get_writer(dest+'.gif', mode='I') as writer:
            for f in fnames_sorted:
                image = imageio.imread(src+f)
                writer.append_data(image)
        
        return