dataset_paths = {
	'celeba_test': 'CelebA_test/',
	'ffhq': 'FFHQ/',
}

model_paths = {
	'pretrained_psp': 'pretrained_models/psp_ffhq_encode.pdparams',
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pdparams',
	'ir_se50': 'pretrained_models/model_ir_se50.pdparams',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'age_predictor':'pretrained_models/dex_age_classifier.pdparams',
	
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pdparams',
	'alexnet': 'pretrained_models/alexnet.pdparams',
	'lin_alex0.1': 'pretrained_models/lin_alex.pdparams',
	'mtcnn_pnet': 'models/mtcnn/mtcnn_paddle/src/weights/pnet.npy',
	'mtcnn_rnet': 'models/mtcnn/mtcnn_paddle/src/weights/rnet.npy',
	'mtcnn_onet': 'models/mtcnn/mtcnn_paddle/src/weights/onet.npy',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}
