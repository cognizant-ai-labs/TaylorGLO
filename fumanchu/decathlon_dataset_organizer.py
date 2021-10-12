
# STANDALONE SCRIPT TO MAKE THE DECATHLON DATASETS USABLE WITHOUT THE ANNOTATION JSON FILES

import os
import json as jsonlib

#config = [
#	['/Users/santiagogonzalez/Downloads/fgvc-aircraft-2013b/data/images_variant_test.txt', '/Users/santiagogonzalez/Downloads/decathlon-1.0-data/aircraft/test'],
#	['/Users/santiagogonzalez/Downloads/fgvc-aircraft-2013b/data/images_variant_train.txt', '/Users/santiagogonzalez/Downloads/decathlon-1.0-data/aircraft/train'],
#	['/Users/santiagogonzalez/Downloads/fgvc-aircraft-2013b/data/images_variant_val.txt', '/Users/santiagogonzalez/Downloads/decathlon-1.0-data/aircraft/val']
#]
#
#for split in config:
#	print("NEW SPLIT: " + split[0])
#	for line in open(split[0]):
#		cols = line.split()
#		if len(cols) > 1:
#			im = cols[0]
#			klass = cols[1]
#
#			path = split[1] + '/' + im + '.jpg'
#			print(path)



config = [
	# ['/Users/santiagogonzalez/Downloads/decathlon-1.0/annotations/aircraft_test_stripped.json', '/Users/santiagogonzalez/Downloads/decathlon-1.0-data/aircraft/test'],
	['/Users/santiagogonzalez/Downloads/decathlon-1.0/annotations/aircraft_train.json', '/Users/santiagogonzalez/Downloads/decathlon-1.0-data/aircraft/train'],
	['/Users/santiagogonzalez/Downloads/decathlon-1.0/annotations/aircraft_val.json', '/Users/santiagogonzalez/Downloads/decathlon-1.0-data/aircraft/val']
]


for split in config:
	print("NEW SPLIT: " + split[0])
	json = jsonlib.loads(open(split[0]).read())

	image_dict = {}
	for image_json in json['images']:
		# if image_json['width'] < 96:
			# print(str(image_json['width']) + " " + str(image_json['height']))
		image_dict[image_json['id']] = image_json['file_name'].split('/')[-1]

	klasses = {}
	for sample in json['categories']:
		ident = sample['id']
		klass = sample['name']
		klasses[ident] = klass.replace('/','_')
		os.system('mkdir ' + split[1] + '/' + klasses[ident])

	print(klasses)

	for sample in json['annotations']:
		ident = sample['image_id']
		klass = klasses[sample['category_id']]

		path = split[1] + '/' + str(ident) + '.jpg'
		newdir = split[1] + '/' + klass
		os.system('mv ' + path + ' ' + newdir)







	# print(len(klasses.keys()))

	# for key in klasses.keys():
	# 	print("CLASS: " + key)
	# 	print("- instances: " + str(len(klasses[key])))
