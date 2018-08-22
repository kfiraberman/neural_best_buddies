import os
import numpy as np
from models import vgg19_model
from algorithms import neural_best_buddies as NBBs
from util import util
from util import MLS

from options.options import Options
opt = Options().parse()

vgg19 = vgg19_model.define_Vgg19(opt)
save_dir =  os.path.join(opt.results_dir, opt.name)

nbbs = NBBs.sparse_semantic_correspondence(vgg19, opt.gpu_ids, opt.tau, opt.border_size, save_dir, opt.k_per_level, opt.k_final, opt.fast)
A = util.read_image(opt.datarootA, opt.imageSize)
B = util.read_image(opt.datarootB, opt.imageSize)
points = nbbs.run(A, B)
mls = MLS.MLS(v_class=np.int32)
mls.run_MLS_in_folder(root_folder=save_dir)
