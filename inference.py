import glob, os
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import cv2
import torch
import argparse

from generate_pairs import Estimator3D

from model.networks import CFRNet
from torchvision.transforms import ToTensor

def normalize(img):
    return (img-0.5)*2

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default="./test_imgs/input", type=str)
    parser.add_argument("--save_path", default="./test_imgs/output", type=str)
    parser.add_argument("--aligner", default=None, type=str, help="if you need to align images, set retinaface")
    parser.add_argument("--generator_path", default="saved_models/CFRNet_G_ep55_vgg.pth", type=str, help="Generator model path")
    parser.add_argument("--estimator_path", default="saved_models/trained_weights_occ_3d.pth", type=str, help="3D estimator model path")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--cuda_id", default=0, type=int)
    args = parser.parse_args()
    
    img_list = glob.glob(f"{args.img_path}/*.jpg")

    det_net = None
    if args.aligner is not None:
        from retinaface.models.retinaface import RetinaFace, load_model
        from retinaface.data import cfg_re50
        det_net = RetinaFace(cfg=cfg_re50, phase='test')
        det_net = load_model(det_net, 'retinaface/weights/Resnet50_Final.pth', False)
        det_net.eval()
        det_net = det_net.cuda()
        print('Finished loading model!')
    
    cudnn.benchmark = True
        
    estimator3d = Estimator3D(render_size=224, model_path=args.estimator_path, det_net=det_net, cuda_id=args.cuda_id)
    cfrnet = CFRNet().cuda()

    trained_weights = torch.load(args.generator_path)
    own_state = cfrnet.state_dict()

    for name, param in trained_weights.items():
        own_state[name[7:]].copy_(param)

    cfrnet.eval()

    for k in tqdm(range(0, len(img_list), args.batch_size)):
        until = k+args.batch_size
        if until > len(img_list):
            until = len(img_list)
        
        # input_img: BGR, cropped: RGB
        input_img = estimator3d.align_convert2tensor(img_list[k:until], aligned=(args.aligner is None))
        # rotated: RGB, guidance: BGR
        rotated, guidance = estimator3d.generate_testing_pairs(input_img, pose=[5.0, 7.0, 0.0])

        rotated = normalize(rotated[...,[2,1,0]].permute(0,3,1,2).contiguous())
        guidance = normalize(guidance[...,[2,1,0]].permute(0,3,1,2).contiguous())
        output, occ_mask = cfrnet(rotated, guidance)
        output = (output / 2) + 0.5
        output = (output.permute(0,2,3,1)*255).cpu().detach().numpy().astype('uint8')

        for i in range(rotated.shape[0]):
            cv2.imwrite(os.path.join(args.save_path, os.path.basename(img_list[k*args.batch_size+i])), cv2.cvtColor(output[i], cv2.COLOR_RGB2BGR))

