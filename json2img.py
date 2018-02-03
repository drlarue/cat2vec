"""Fetch pictures for Petfinder listings"""

import os, errno
import sys
import glob

import json
#from pprint import pprint
import urllib.request


def json2pic(filepath, download_to):
    """Download images for each Petfinder listing

    Arguments:
        * filepath: json file from the Petfinder API call
        * download_to: output directory for images
    """
    input_file = open(filepath, 'r')
    json_decode = json.load(input_file)

    no_pets = len(json_decode['petfinder']['pets']['pet'])

    mydict = {}
    for i in range(no_pets):
        for ll in json_decode['petfinder']['pets']['pet'][i]['media']:
            if ll == 'photos':
                # include only the listings that have pics
                for key, value in json_decode['petfinder']['pets']['pet'][i]['id'].items():
                    mydict[value] = {}
                    for l in json_decode['petfinder']['pets']['pet'][i]['media']['photos']['photo']:
                        if l['@size'] == 'x':
                            mydict[value][l['@id']] = l['$t']

    for key, value in mydict.items():
        for nested_key, url in value.items():

            if not os.path.exists(download_to):
                os.makedirs(download_to)

            path = os.path.join(download_to + key + '_' + nested_key + 'jpeg')

            try:
                urllib.request.urlretrieve(url, path)
            except OSError:
                print(key, nested_key)
