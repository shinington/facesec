"""
Configuration of the experiments.
"""

config = {
	'pgd': {
	'epsilon': 8/255.,
	'alpha': 2/255.,
	'step': 40
	},
	'occlusion': {
	'epsilon': 255/255.,
	'alpha': 4/255.,
	'step': 200
	},
	'facemask': {
	'epsilon': 255/255.,
	'alpha': 4/255.,
	'step': 200,
	'width': 16, # width of color grids on face masks
	'height': 8 # heights of color grids on face masks
	},
	'threshold': {
	'vggface': 0.6408,
	'facenet': 0.4338
	},
	'batch_size': 50
}