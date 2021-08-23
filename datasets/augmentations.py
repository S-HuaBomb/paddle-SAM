import numpy as np
import paddle

class AgeTransformer(object):
	def __init__(self, target_age):
		self.target_age = target_age
	
	def __call__(self, img):
		img = self.add_aging_channel(img)
		return img
	
	def add_aging_channel(self, img):
		target_age = self.__get_target_age()
		target_age = int(target_age) / 100
		img = paddle.concat((img, target_age*paddle.ones((1, img.shape[1], img.shape[2]))))
		return img
	
	def __get_target_age(self):
		if self.target_age == "uniform_random":
			return np.random.randint(low=0., high=101, size=1)[0]
		else:
			return self.target_age
