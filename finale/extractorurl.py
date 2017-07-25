from urllib.parse import urlparse
import csv
from ast import literal_eval as make_tuple
import re
import os


'''
extracts features for URL model
'''

def getUrlPath(url):
    parsed = urlparse(url)
    path = parsed.path
    return path


def getUrlQuery(url):
	parsed = urlparse(url)
	query = parsed.query
	if query:
		return query
	else:
		return ' '


def getUrlFilename(url):
	parsed = urlparse(url)
	filename = parsed.path.split('/')
	for f in filename:
		if '.' in f:
			return f
	return ' '


def absolute(urls):
    A = set()
    for url in urls:
        A.add(len(getUrlPath(url)))

    aList = [e for e in A]
    aList.sort()
    start = 0
    end = 0
    ranges = []

    if len(aList) == 1:
        ranges.append((aList[0],aList[0]))
    else:
        for i,e in enumerate(aList):
            if i == 0:
                end = e
                start = e
            else:
                end += 1
                if e != end:
                    end = end - 1
                    if (end-start) > 2:
                        ranges.append(str((start,end)),)
                    start = e
                    end = e
                elif i == len(aList)-1:
                    if (end-start) > 2:
                        ranges.append(str((start, end)), )
    return ranges


def nesting(urls):
    A = set()
    for url in urls:
        summe = nestingCount(url)
        A.add(summe)
    aList = [e for e in A]
    aList.sort()
    start = 0
    end = 0
    ranges = []
    if len(aList) == 1:
        ranges.append((aList[0],aList[0]))
    else:
        for i, e in enumerate(aList):
            if i == 0:
                end = e
                start = e
            else:
                end += 1
                if e != end:
                    end = end - 1
                    ranges.append(str((start, end)), )
                    start = e
                    end = e
                elif i == len(aList)-1:
                    ranges.append(str((start, end)), )
    return ranges


def nestingCount(url):
    path = getUrlPath(url)
    summe = 0
    if path:
        if path[len(path) - 1] == '/':
            path = path[0:len(path) - 1]
        s = path.split('/')
        summe = sum(1 for e in s if e)
    return summe

def average(urls):
    summe = 0.0
    count = 0.0
    for url in urls:
        summe += len(url)
        count += 1.0
    row = []
    row.append(summe/count)
    return row
