from os import listdir

labels_file = '../labels.txt'
paths = [('aic540', 'train'), ('aic540', 'val'), ('aic480', 'train'), ('aic480', 'val')]
labels_dir = '/datasets/{}/{}/labels/'
images_dir = '/datasets/{}/{}/images/'

with open(labels_file, 'w') as f_new_labels:
	for dataset, subset in paths:
		labels = labels_dir.format(dataset, subset)
		imgs = images_dir.format(dataset, subset)
		print "Processing labels:", labels
		label_list = listdir(labels)
		for i, img in enumerate(label_list):
			if i % 1000 == 0:
				print "\tProcessing image {} of {}".format(i, len(label_list))
			with open(labels+img, 'r') as f_old_labels:
				img_labels = f_old_labels.readlines()
				for line in img_labels:
					elements = line.split()
					l_class = elements.pop(0)
					elements.insert(0, imgs+img.strip('.txt')+'.jpeg')
					elements.append(l_class)
					new_line = ','.join(elements)
					f_new_labels.write(new_line+'\n')
