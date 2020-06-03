import os
import cv2

# save_path = '/home/data2/DATASET/test_set_a/test_set_a/mask'
save_path = '/home/data2/DATASET/vschallenge/test/mask_0525'

read_path = '/home/data2/miles/HRNet_Parsing/res/test_results'


length = len(os.listdir(read_path))
print(length)

for name in os.listdir(read_path):
	# print(name)
	if name[-3:] == 'png':
		dir_name = name[:-9]
		# print(dir_name)
		img = cv2.imread(os.path.join(read_path, name))

		if not os.path.exists(save_path+'/'+dir_name):
			os.makedirs(save_path+'/'+dir_name)
		cv2.imwrite(os.path.join(save_path, dir_name, name), img)

		print(name+' finished.')

