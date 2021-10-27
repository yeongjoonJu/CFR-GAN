## Complete Face Recovery GAN: Unsupervised Joint Face Rotation and De-Occlusion from a Single-View Image (WACV 2022)
> Yeong-Joon Ju, Gun-Hee Lee, Jung-Ho Hong, and Seong-Whan Lee

Below images are frontalization and de-occlusion results. The first rows are the input images and the second rows are our results. We crop the results as alignments for input images of facial recognition networks.

![](figure/app_samples.png)

### Abstract
Although various face-related tasks have significantly advanced in recent years, occlusion and extreme pose still impede the achievement of higher performance. Existing face rotation or de-occlusion methods only have emphasized the aspect of each problem. In addition, the lack of high-quality paired data remains an obstacle for both methods. In this work, we present a self-supervision strategy called Swap-R&R to overcome the lack of ground-truth in a fully unsupervised manner for joint face rotation and deocclusion. To generate an input pair for self-supervision, we transfer the occlusion from a face in an image to an estimated 3D face and create a damaged face image, as if rotated from a different pose by rotating twice with the roughly de-occluded face. Furthermore, we propose Complete Face Recovery GAN (CFR-GAN) to restore the collapsed textures and disappeared occlusion areas by leveraging the structural and textural differences between two rendered images. Unlike previous works, which have selected occlusion-free images to obtain ground-truths, our approach does not require human intervention and paired data. We show that our proposed method can generate a de-occluded frontal
face image from an occluded profile face image. Moreover, extensive experiments demonstrate that our approach can boost the performance of facial recognition and facial expression
recognition.

**\*\*The codes implemented for Swap-R&R and CFR-GAN will be updated later.\*\***

Training codes for occlusion robust 3D face reconstruction in this paper are available in [here](https://github.com/yeongjoonJu/Occlusion-Robust-3D-Face-CFR-GAN).
