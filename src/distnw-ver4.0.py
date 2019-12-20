# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:57:15 2019

Author: Rounak Meyur
Description: This program reads the sample Blacksburg power network from the pdf file
and converts them to png image. Then it identifies the edges in the network and tries
to generate the corresponding networkx graph.
"""

import sys,os
import matplotlib.pyplot as plt
# import networkx as nx
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (PDFInfoNotInstalledError,
                                  PDFPageCountError,PDFSyntaxError)

workPath = os.getcwd()
inpPath = workPath + "/input/AEP Files/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
figPath = workPath + "/figs/AEP Images/"
tmpPath = workPath + "/temp/AEP PPM/"

sys.path.append(libPath)

files = os.listdir(path=inpPath)[:-1]
for f in files:
    out_file = f.split('.')[0]+'.jpg'
    images = convert_from_path(inpPath+f,output_folder=tmpPath)
    for image in images:
        image.save(figPath+out_file,'JPEG')