import numpy as np
import torch

cfgs = [
    {'feature_map': 38,
     'min_size': 21,
     'max_size': 45,
     'feature_map_shrink': 8,
     'aspect_ratios': [2]
        },
    {'feature_map': 19,
     'min_size': 45,
     'max_size': 99,
     'feature_map_shrink': 16,
     'aspect_ratios': [2, 3]
        },
    {'feature_map': 10,
     'min_size': 99,
     'max_size': 153,
     'feature_map_shrink': 32,
     'aspect_ratios': [2, 3]
        },
    {'feature_map': 5,
     'min_size': 153,
     'max_size': 207,
     'feature_map_shrink': 64,
     'aspect_ratios': [2, 3]
        },
    {'feature_map': 3,
     'min_size': 207,
     'max_size': 261,
     'feature_map_shrink': 100,
     'aspect_ratios': [2]
        },
    {'feature_map': 1,
     'min_size': 261,
     'max_size': 315,
     'feature_map_shrink': 300,
     'aspect_ratios': [2]
        }
    
    
      ]

def generate_priors(cfgs, clip=True, image_size=300):
	"""
	Generate prior boxes

	Args:
		cfgs(list): list of dictionaries with feature map size, minimum scale of prior boxes, maximum scale of prior boxes, and aspect ratio
		clip(bool): clip the values between 0 and 1
		image_size(int): size of an input image

	Returns:
		priors(torch.Tensor): returns prior boxes for every feature map in cfgs
	"""
    priors = []
    for cfg in cfgs:
        f_k = image_size / cfg['feature_map_shrink'] # k-th feature map size
        
        for i in range(cfg['feature_map']):
            for j in range(cfg['feature_map']):
                # center of each deafult box(relative to unit box)
                center_x = (j + 0.5) / f_k
                center_y = (i + 0.5) / f_k
                
                # square box (ratio 1, box size = min_size)
                s_k = h = w = cfg['min_size'] / image_size # divide by image size to make unit box
                priors.append([center_x, center_y, h, w])
    
                # square box (ratio 1, box size = sqrt(s_k * s_(k+1))   )
                h = w = np.sqrt(s_k * (cfg['max_size']/ image_size))
                priors.append([center_x, center_y, h, w])
                # ratio (2 or 3)
                
                for ar in cfg['aspect_ratios']:
                    priors.append([center_x, center_y, s_k *np.sqrt(ar), s_k/np.sqrt(ar)])
                    priors.append([center_x, center_y, s_k/np.sqrt(ar), s_k*np.sqrt(ar)])
    priors = torch.Tensor(priors)
    
    if clip:
        priors.clamp_(max=1, min=0)
        
    return priors

