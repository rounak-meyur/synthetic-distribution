# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:57:15 2019

Author: Rounak Meyur
Description: This program reads the sample Blacksburg power network from the pdf file
and converts them to png image. Then it identifies the edges in the network and tries
to generate the corresponding networkx graph.
"""

import sys,os
from bs4 import BeautifulSoup
import csv
import matplotlib.pyplot as plt
# import networkx as nx
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (PDFInfoNotInstalledError,
                                  PDFPageCountError,PDFSyntaxError)

workPath = os.getcwd()
inpPath = workPath + "/input/AEP Files/"
libPath = workPath + "/Libraries/"
csvPath = workPath + "/csv/"
kmlPath = workPath + "/kml/"
figPath = workPath + "/figs/AEP Images/"
tmpPath = workPath + "/temp/AEP PPM/"

sys.path.append(libPath)


def convert_pdf2jpeg(inpPath,tmpPath,figPath):
    """
    """
    files = os.listdir(path=inpPath)[:-1]
    for f in files:
        out_file = f.split('.')[0]+'.jpg'
        images = convert_from_path(inpPath+f,output_folder=tmpPath)
        for image in images:
            image.save(figPath+out_file,'JPEG')
    return

def process_coordinate_string(string):
    """
    """
    space_splits = string.replace('\t','').replace('\n','').strip().split(" ")
    ret = []
    for split in space_splits:
        comma_split = split.split(',')
        ret.append(','.join([comma_split[0],comma_split[1]]))
    return ret



def convert_kml2csv(kmlPath,kmlfile,csvPath):
    """
    """
    prefix = kmlfile[5:]
    data = []
    edges = []
    # open kml file and read the list of coordinates
    with open(kmlPath+kmlfile+'.kml','r') as f:
        s = BeautifulSoup(f,'xml')
        start = 0
        for coords in s.find_all('coordinates'):
            ret = process_coordinate_string(coords.string)
            data.extend(ret)
            edges.extend([prefix+str(start+i)+','+prefix+str(start+i+1) for i in range(len(ret)-1)])
            start += len(ret)
        data = [prefix+str(i)+','+data[i] for i in range(len(data))]
    # open csv file and list down the coordinates with generated IDs
    with open(csvPath+kmlfile+'.csv','w') as csvfile:
        csvfile.write('\n'.join(data))
    with open(csvPath+kmlfile+'-edges.csv','w') as csvfile:
        csvfile.write('\n'.join(edges))
    return


convert_kml2csv(kmlPath,'Path_290C1',csvPath)






















