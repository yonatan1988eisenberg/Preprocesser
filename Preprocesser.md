# Preprocesser

## Introduction

> Preprocesser is a python program aimed to assist in the process of data cleaning before it can be uploaded to Bridgify's production database.

## Code Samples

> Preprocesser receives a csv file which contains a list of items and returns an excel file. The first sheet in this file is the original csv and the others sheets contain clusters of items which are suspected to be related to one another.
-p C:/Users/Yonatan/Bridgify_Internship/Preprocesser/new_york_google.csv -s testing.xlsx

> Input Parameters:
--path \ -p (required) : the path to the input csv file.  <br>
--save \ -s : the path for the output xlsx file. If only a filename is provided it will be saved in the input file's folder.
Only use '/' (and not '\\') for the input parameters

## Installation

> To run this program python 3.9 is required. Additional packages are described in the requirements.txt file
 
## License
[MIT](https://choosealicense.com/licenses/mit/)