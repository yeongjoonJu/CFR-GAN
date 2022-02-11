## Complete Face Recovery GAN: Unsupervised Joint Face Rotation and De-Occlusion from a Single-View Image (WACV 2022)
> [Yeong-Joon Ju](https://github.com/yeongjoonJu), Gun-Hee Lee, Jung-Ho Hong, and Seong-Whan Lee

[[Video]](https://www.youtube.com/watch?v=NVm4MzqvgO0) [[Paper]](https://openaccess.thecvf.com/content/WACV2022/html/Ju_Complete_Face_Recovery_GAN_Unsupervised_Joint_Face_Rotation_and_De-Occlusion_WACV_2022_paper.html)

Below images are frontalization and de-occlusion results. The first rows are the input images and the second rows are our results. We crop the results as alignments for input images of facial recognition networks.

<img src="figure/app_samples.png" style="zoom:60%;" />

### Abstract

We present a self-supervision strategy called Swap-R&R to overcome the lack of ground-truth in a fully unsupervised manner for joint face rotation and de-occlusion. To generate an input pair for self-supervision, we transfer the occlusion from a face in an image to an estimated 3D face and create a damaged face image, as if rotated from a different pose by rotating twice with the roughly de-occluded face. Furthermore, we propose Complete Face Recovery GAN (CFR-GAN) to restore the collapsed textures and disappeared occlusion areas by leveraging the structural and textural differences between two rendered images. Unlike previous works, which have selected occlusion-free images to obtain ground-truths, our approach does not require human intervention and paired data.

<img src="figure/stages.png" style="zoom:80%;" />

**\*\*README and trained weights will be updated soon.\*\***

Training codes for occlusion robust 3D face reconstruction in this paper are available in [here](https://github.com/yeongjoonJu/Occlusion-Robust-3D-Face-CFR-GAN).
