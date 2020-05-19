# Neural Best-Buddies in PyTorch

This is our PyTorch implementation for the Neural-Best Buddies paper.

The code was written by [Kfir Aberman](https://kfiraberman.github.io/) and supported by [Mingyi Shi](https://rubbly.cn/).

**Neural Best-Buddies: [Project](https://kfiraberman.github.io/neural_best_buddies/) |  [Paper](https://arxiv.org/pdf/1805.04140.pdf)**
<img src="./images/teaser.jpg" width="800" />

If you use this code for your research, please cite:

Neural Best-Buddies: Sparse Cross-Domain Correspondence
[Kfir Aberman](https://kfiraberman.github.io/), [Jing Liao](https://liaojing.github.io/html/), [Mingyi Shi](https://rubbly.cn/), [Dani Lischinski](http://danix3d.droppages.com/), [Baoquan Chen](http://www.cs.sdu.edu.cn/~baoquan/), [Daniel Cohen-Or](https://www.cs.tau.ac.il/~dcor/), SIGGRAPH 2018.

## Prerequisites
- Linux or macOS
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN
- Pytorch > (1.x.x)

### Run

- Run the algorithm (demo example)
```bash
#!./script.sh
python3 main.py --datarootA ./images/original_A.png --datarootB ./images/original_B.png --name lion_cat --k_final 10
```
The option `--k_final` dictates the final number of returned points. The results will be saved at `../results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.

### Output
Sparse correspondence:
- correspondence_A.txt, correspondence_B.txt
- correspondence_A_top_k.txt, correspondence_B_top_k.txt

Dense correspondence (densifying based on [MLS](http://faculty.cse.tamu.edu/schaefer/research/mls.pdf)):
-  BtoA.npy, AtoB.npy

Warped images (aligned to their middle geometry):
- warp_AtoM.png, warp_BtoM.png

### Tips
- If you are running the algorithm on a bunch of pairs, we recommend to stop it at the second layer to reduce runtime (comes at the expense of accuracy), use the option `--fast`.
- If the images are very similar (e.g, two frames extracted from a video), many corresponding points might be found, resulting in long runtime. In this case we suggest to limit the number of corresponding points per level by setting `--k_per_level 20` (or any other desired number)

## Image Morphing

The morphing results in the paper are based on the paper [Automating image morphing using structural similarity on a halfway domain](http://hhoppe.com/morph.pdf) by Liao et al.

Here are detailed instructions for how to combine the output of our code with the image morphing code:

First, please download the exe file from [here](https://drive.google.com/drive/folders/0BwMKxLMS8dFBSTBPa2lRUWxGbFk) and test whether it works on your machine with the given demo case as described in the UserGuide.dox or simply run “runme.bat”. Alternatively, the source code can be found [here](https://github.com/liaojing/Image-Morphing). When it works:

1. Make a folder e.g. “case1”, containing three files in it: the two input images (should be with the same size) called image1.png and image2.png respectively, and a setting file called “settings.xml”. You can copy the setting.xml form “baby” and modify it to the points that were extracted by our code. For example:

`<points image1="0.273438 0.643973 0.657366 0.561384 0.791295 0.641741 0.768973 0.757812 0.309152 0.989955 0.775442 0.994469 0.373051 0.620267 0.420898 0.452148 " image2="0.237832 0.686947 0.627212 0.448009 0.755531 0.625000 0.722345 0.790929 0.308628 0.990044 0.775442 0.994469 0.339644 0.586860 0.420898 0.452148 "/>`

   where the format is:

`<points image1="x1(in image1) y1(in image1)  x2(in image1)  y2(in image1)  x3(in image1)  y3(in image1)  ……" image2=" x1(in image2) y1(in image2)  x2(in image2)  y2(in image2)  x3(in image2)  y3(in image2)  "/>`

2. Make a subfolder called “0” under the main folder “case1”. Put two mask images “mask1.png” “mask2.png”  that have the same size as the input images and set all the 3 channels to zero.

3. Run `ImageMorphing.exe case1\settings.xml -auto`

## Citation
If you use this code for your research, please cite our paper:
```
@article{aberman2018neural,
  title={Neural best-buddies: Sparse cross-domain correspondence},
  author={Aberman, Kfir and Liao, Jing and Shi, Mingyi and Lischinski, Dani and Chen, Baoquan and Cohen-Or, Daniel},
  journal={ACM Transactions on Graphics (TOG)},
  volume={37},
  number={4},
  pages={69},
  year={2018},
  publisher={ACM}
}
```
