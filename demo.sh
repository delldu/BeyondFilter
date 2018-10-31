#/************************************************************************************
#***
#***    File Author: Dell, 2018-10-31 13:39:02
#***
#************************************************************************************/
#
#! /bin/sh

python gauss.py images/gauss_girl.jpg 
python guided.py images/guided_girl.jpg
python guided.py images/enhance.jpg
python dehaze.py images/haze.jpg
