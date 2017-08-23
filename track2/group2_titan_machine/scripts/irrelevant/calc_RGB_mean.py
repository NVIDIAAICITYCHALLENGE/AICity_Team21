from scipy.misc import imread
import numpy as np
import os


"""
 List of directories where images are. 
"""
src_directories = [
	'/datasets/aic540/train/images/',
	'/datasets/aic540/val/images/',
	'/datasets/aic480/train/images/',
	'/datasets/aic480/val/images/'
]


"""
 Returns a 3-tuple that is the mean RGB values of the provided image.
"""
def get_RGB_mean(image):
	total = float(image.shape[0]) / 3.
	r, g, b = 0., 0., 0.
	for i,p in enumerate(image):
		if i % 3 == 0:
			r += p
		elif (i-1) % 3 == 0:
			g += p
		else:
			b += p
	return r/total, g/total, b/total


"""
 Iterate over all the image directories and compute the running average of RGB
 values across all of images in these directories.
"""
r_avg, g_avg, b_avg, total = 0., 0., 0., 0.
for src in src_directories:
	print "Analyzing dataset {}".format(src)
	images = os.listdir(src)
	total += len(images)
	for i, img in enumerate(images):
		if i % 1000 == 0:
			print "\tAnalyzing image {} of {}".format(i, len(images))
		r, g, b = get_RGB_mean(imread(src+img).flatten())
		r_avg += r
		g_avg += g
		b_avg += b	

r_avg, g_avg, b_avg = r_avg/total, g_avg/total, b_avg/total
print "The overall RGB mean values are {}".format((r_avg,g_avg,b_avg))
with open('../model/RGB_mean.txt', 'w') as f:
	f.write((r_avg, g_avg, b_avg))
