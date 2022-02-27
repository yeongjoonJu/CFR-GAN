#!/usr/bin/python
# -*- encoding: utf-8 -*-
from ctypes import ArgumentError
import os ; import sys 
os.chdir( os.path.split( os.path.realpath( sys.argv[0] ) )[0] ) 

from mmRegressor.network.resnet50_task import *
from mmRegressor.preprocess_img import Preprocess
from mmRegressor.load_data import *
from mmRegressor.reconstruct_mesh import Reconstruction, Compute_rotation_matrix, _need_const, Projection_layer

import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2, glob
# from torchsummary import summary
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    cameras, lighting,
    PointLights, HardPhongShader,
    RasterizationSettings,
    BlendParams,
    MeshRenderer, MeshRasterizer
)
from tqdm import tqdm
from tools.ops import erosion, SCDiffer, dilation, blur

# Retina Face
if os.path.exists('Pytorch_Retinaface'):
    from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
    from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
    from Pytorch_Retinaface.utils.box_utils import decode, decode_landm


class Estimator3D(object):
    def __init__(self, is_cuda=True, batch_size=1, render_size=224, test=True, model_path=None, back_white=False, cuda_id=0, det_net=None):
        self.is_cuda = is_cuda
        self.render_size = render_size
        self.cuda_id = cuda_id
        # Network, cfg
        if det_net is not None:
            self.det_net = det_net[0]
            self.det_cfg = det_net[1]

        # load models
        if model_path is None:
            print('Load pretrained weights')
        else:
            print('Load {}'.format(model_path))
        self.load_3dmm_models(model_path, test)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])
                
        self.argmax = lambda i, c: c[i]
        self.thresholding = torch.nn.Threshold(0.3, 0.0)

        tri = self.face_model.tri
        tri = np.expand_dims(tri, 0)
        self.tri = torch.FloatTensor(tri).repeat(batch_size, 1, 1)

        self.skin_mask = -1*self.face_model.skin_mask.unsqueeze(-1)
        
        if is_cuda:
            device = torch.device('cuda:'+str(cuda_id))
            self.tri = self.tri.cuda(cuda_id)
        else:
            device = torch.device('cpu')

        # Camera and renderer settings
        blend_params = BlendParams(background_color=(0.0,0.0,0.0))
        if back_white:
            blend_params= BlendParams(background_color=(1.0,1.0,1.0))

        self.R, self.T = look_at_view_transform(eye=[[0,0,10]], at=[[0,0,0]], up=[[0,1,0]], device=device)
        camera = cameras.FoVPerspectiveCameras(znear=0.01, zfar=50.0, aspect_ratio=1.0, fov=12.5936, R=self.R, T=self.T, device=device)
        lights = PointLights(ambient_color=[[1.0,1.0,1.0]], device=device, location=[[0.0,0.0,1e-5]])
        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=RasterizationSettings(
                    image_size=render_size,
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    cull_backfaces=True
                )
            ),
            shader=HardPhongShader(cameras=camera, device=device, lights=lights, blend_params=blend_params)
        )

    def load_3dmm_models(self, model_path, test=True):
        # read face model
        self.face_model = BFM('mmRegressor/BFM/BFM_model_80.mat', self.cuda_id)

        # read standard landmarks for preprocessing images
        self.lm3D = self.face_model.load_lm3d("mmRegressor/BFM/similarity_Lm3D_all.mat")
        
        regressor = resnet50_use()
        if model_path is None:
            regressor.load_state_dict(torch.load("mmRegressor/network/th_model_params.pth"))
        else:
            regressor.load_state_dict(torch.load(model_path, map_location='cuda:'+str(self.cuda_id)))

        if test:
            regressor.eval()
        if self.is_cuda:
            regressor = regressor.cuda(self.cuda_id)
        if test:
            for param in regressor.parameters():
                param.requires_grad = False

        self.regressor = regressor

    def regress_3dmm(self, img):
        arr_coef = self.regressor(img)
        coef = torch.cat(arr_coef, 1)

        return coef


    def reconstruct(self, coef, test=False):
        # reconstruct 3D face with output coefficients and face model
        face_shape, _, face_color, _,face_projection,_,gamma = Reconstruction(coef,self.face_model)
        verts_rgb = face_color[...,[2,1,0]]
        mesh = Meshes(verts=face_shape, faces=self.tri[:face_shape.shape[0],...], textures=Textures(verts_rgb=verts_rgb))

        rendered = self.phong_renderer(meshes_world=mesh, R=self.R, T=self.T)
        rendered = torch.clamp(rendered, 0.0, 1.0)

        landmarks_2d = torch.zeros_like(face_projection).cuda(self.cuda_id)
        landmarks_2d[...,0] = torch.clamp(face_projection[...,0].clone(), 0, self.render_size-1)
        landmarks_2d[...,1] = torch.clamp(face_projection[...,1].clone(), 0, self.render_size-1)
        landmarks_2d[...,1] = self.render_size - landmarks_2d[...,1].clone() - 1
        landmarks_2d = landmarks_2d[:,self.face_model.keypoints,:]

        if test:
            return rendered, landmarks_2d

        tex_mean = torch.sum(face_color*self.skin_mask) / torch.sum(self.skin_mask)
        ref_loss = torch.sum(torch.square((face_color - tex_mean)*self.skin_mask)) / (face_color.shape[0]*torch.sum(self.skin_mask))

        gamma = gamma.view(-1,3,9)    
        gamma_mean = torch.mean(gamma, dim=1, keepdim=True)
        gamma_loss = torch.mean(torch.square(gamma - gamma_mean))

        return rendered, landmarks_2d, ref_loss, gamma_loss


    def estimate_and_reconstruct(self, img):
        coef = self.regress_3dmm(img)
        return self.reconstruct(coef, test=True)

    
    def estimate_five_landmarks(self, img):
        # Detect and align
        img_raw = np.array(img)
        ori_h, ori_w, _ = img_raw.shape
        img = np.float32(cv2.resize(img_raw, (320, 320), cv2.INTER_CUBIC))
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.cuda_id)
        scale = scale.to(self.cuda_id)
        
        loc, conf, landms = self.det_net(img)
        priorbox = PriorBox(self.det_cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.cuda_id)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.det_cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.det_cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.cuda_id)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > 0.1)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        # order = scores.argsort()[::-1][:args.top_k]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
        landms = landms[keep]
        landm = np.array(landms[0])
        
        landm = np.reshape(landm, (-1, 2))
        landm[:,0] *= ori_w/320
        landm[:,1] *= ori_h/320

        return landm


    def align_convert2tensor(self, img_list, aligned=False):
        if not aligned and self.det_net is None:
            raise ArgumentError('Detection network is None!')

        input_img = []
        for filename in img_list:
            if aligned:
                img = cv2.imread(filename)
                if img.shape[0]!= self.render_size:
                    img = cv2.resize(img, (self.render_size,self.render_size), cv2.INTER_AREA)
                if img.shape[2]==4:
                    img = img[...,:3]
            else:
                img = Image.open(filename)
                lm = self.estimate_five_landmarks(img)
                img, _ = Preprocess(img, lm, self.lm3D, render_size=self.render_size)
                img = img[0].copy()
            
            img = self.to_tensor(img)
            input_img.append(img.unsqueeze(0))

        input_img = torch.cat(input_img)
        if self.is_cuda:
            input_img = input_img.type(torch.FloatTensor).cuda(self.cuda_id)
        
        return input_img
    
    
    def get_occlusion_mask(self, model, scd, rot, gui, ori_img, obj=False):
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        std = torch.FloatTensor([0.229, 0.224, 0.225]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # skin 1, nose 2, eye_glass 3, r_eye 4, l_eye 5, r_brow 6, l_brow 7, r_ear 8, l_ear 9,
        # inner_mouth 10, u_lip 11, l_lip 12, hair 13
        parsed = model(F.interpolate((ori_img-mean) / std, (512,512), mode='bilinear', align_corners=True))
        parsed = torch.argmax(F.interpolate(parsed, (self.render_size, self.render_size), mode='bilinear', align_corners=True), dim=1, keepdim=True)

        guidance_gray = torch.mean(gui, dim=1, keepdim=True)
        guidance_noise = (guidance_gray < 0.04)

        coarse_occ_mask = torch.ones_like(guidance_noise, dtype=rot.dtype).cuda()
        idx = (parsed==1).type(torch.BoolTensor) | (parsed==2).type(torch.BoolTensor) | ((parsed>=4).type(torch.BoolTensor) & (parsed<=7).type(torch.BoolTensor)) \
            | ((parsed>=10).type(torch.BoolTensor) & (parsed<=12).type(torch.BoolTensor))
        coarse_occ_mask[idx] = 0.0

        eye_idx = (parsed==4).type(torch.cuda.BoolTensor) | (parsed==5).type(torch.cuda.BoolTensor)

        parsed_g = model(F.interpolate((gui-mean) / std, (512,512), mode='bilinear', align_corners=True))
        parsed_g = torch.argmax(F.interpolate(parsed_g, (self.render_size, self.render_size), mode='bilinear', align_corners=True), dim=1, keepdim=True)
        parsed_g = torch.logical_or(parsed_g==4, parsed_g==5)

        closed_eye_batch = torch.ge(torch.sum(torch.logical_and(parsed_g, parsed==1).float(), dim=[1,2,3]) / (torch.sum(parsed_g.float(), dim=[1,2,3]) + 1e-7), 0.5)
        closed_eye_batch = closed_eye_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(eye_idx)
        eye_idx = torch.logical_or(eye_idx, torch.logical_and(parsed_g, closed_eye_batch))

        eye_idx = dilation(eye_idx.type(torch.cuda.FloatTensor), filter_size=5).type(torch.cuda.BoolTensor)
        coarse_occ_mask[eye_idx] = 0.0
        coarse_occ_mask[guidance_noise] = 0.0

        # Remove edge
        background = torch.zeros_like(guidance_noise, dtype=gui.dtype).cuda()
        background[guidance_noise] = 1.0
        background[eye_idx] = 0.0
        background[torch.logical_or(parsed==3, parsed==10)] = 0.0

        gui_with_bg = ori_img*background + gui*(1.-background)

        background = erosion(background, filter_size=2)
        background = blur(background, filter_size=7)
        edge = torch.zeros_like(background).cuda()
        edge[torch.logical_and(background>0.0, background<0.8)] = 1.0

        # Generate eye_mask
        eye_mask = torch.zeros_like(eye_idx, dtype=rot.dtype).cuda()
        eye_mask[eye_idx] = 1.0

        if not obj:
            """
            Occlusions by objects and rotaton
            """
            diff = scd(rot, gui, alpha=0.33) * (1.-eye_mask)
            eye_diff = scd(rot, ori_img, alpha=0.5) * eye_mask
            adaptive_guide = diff + (coarse_occ_mask + eye_diff)
            adaptive_guide = torch.clamp(adaptive_guide, 0.0, 1.0)

            # Remove edge
            adaptive_guide *= (1.-edge)
            adaptive_guide = torch.round(adaptive_guide)
            
            return adaptive_guide
        else:
            """
            Occlusions by only objects
            """
            diff = torch.round(scd(ori_img, gui_with_bg)) * (1.-eye_mask)
            brow_lip_mask = torch.ones_like(eye_idx, dtype=rot.dtype).cuda()
            brow_lip_mask[torch.logical_or(torch.logical_or(parsed==6, parsed==7), torch.logical_or(parsed==11, parsed==12))] = 0.6
            diff[parsed==10] = 0.0
            diff *= brow_lip_mask
                        
            guide_obj = diff + coarse_occ_mask
            guide_obj = torch.clamp(guide_obj, 0.0, 1.0)

            face_mask = torch.sum(guidance_gray >= 0.04, dim=[1,2,3])
            occ_cands = torch.sum(torch.round(guide_obj + 0.1), dim=[1,2,3])
            occ_rates = occ_cands / face_mask
            
            for b in range(occ_rates.shape[0]):
                if occ_rates[b] < 0.09:
                    guide_obj[b] *= 0.1

            return guide_obj
        
        
    def get_colors_from_image(self, image, proj, z_buffer, scaling=True, normalized=False, reverse=True, z_cut=None):
        #image *= mask
        if image.shape[1]==3 or image.shape[1]==4:
            image = image.permute(0,2,3,1)
        _, h, w, _ = image.shape

        if scaling:
            proj *= self.render_size / 224
        
        proj[...,0] = torch.clamp(proj[...,0], 0, w-1) # w
        proj[...,1] = torch.clamp(proj[...,1], 0, h-1)
        if reverse:
            proj[...,1] = h - proj[...,1] - 1
        idx = torch.round(proj).type(torch.long)

        colors = None
        for k in range(len(image)):                   
            if colors is None:
                colors = image[k, idx[k,:,1], idx[k,:,0]].unsqueeze(0)
            else:
                colors = torch.cat([colors, image[k, idx[k,:,1], idx[k,:,0]].unsqueeze(0)])

        z_buffer = z_buffer.squeeze(2)
        
        if z_cut is not None:
            colors = colors.repeat(2,1,1)
            colors[:z_buffer.shape[0]][z_buffer < z_cut] = 0.0

        if not normalized:
            colors = colors/255

        if self.is_cuda:
            colors = colors.cuda(self.cuda_id)

        return colors


    def swap_and_rotate_and_render(self, image, parsing_net, scd):
        # Regress 3DMM parameters
        coef = self.regress_3dmm(image) # RGB2BGR
        # Reconstruct shape and color from 3DMM parameters
        face_shape, ori_angles, translation, face_color, _, face_projection, z_buffer, front_face = Reconstruction(coef,self.face_model)
        # get color value from image
        color_from_img = self.get_colors_from_image(image, face_projection, z_buffer, normalized=True)

        # Mesh with color from image
        rot = Meshes(verts=face_shape, faces=self.tri[:image.shape[0],...], textures=Textures(verts_rgb=color_from_img))
        rot = self.phong_renderer(meshes_world=rot, R=self.R, T=self.T)
        rot = torch.clamp(rot, 0.0, 1.0)
        
        gui = Meshes(verts=face_shape, faces=self.tri[:image.shape[0],...], textures=Textures(verts_rgb=face_color[...,[2,1,0]]))
        gui = self.phong_renderer(meshes_world=gui, R=self.R, T=self.T)
        gui = torch.clamp(gui, 0.0, 1.0)

        rot_ori = rot.permute(0,3,1,2)[:,[2,1,0],...]
        gui_ori = gui.permute(0,3,1,2)[:,[2,1,0],...]

        # Swap
        mask_only_obj = self.get_occlusion_mask(parsing_net, scd, rot, gui_ori, image[:,[2,1,0],...], obj=True)
        blur_mask_o = blur(dilation(mask_only_obj, filter_size=3), filter_size=3)
        rot = image[:,[2,1,0],...]*(1.-blur_mask_o) + gui_ori*blur_mask_o

        gui = gui_ori*(1.-mask_only_obj) + rot_ori*mask_only_obj
        gui = gui*(1.-blur_mask_o) + blur_mask_o*blur(gui, filter_size=3)

        # get color value from image
        color_from_img = self.get_colors_from_image(rot, face_projection, z_buffer, normalized=True, reverse=False, scaling=False)
        
        # Rotate and render
        # get random rotation angles and rotate (x:pitch, y:yaw, z:roll)
        angles = torch.rand(rot.shape[0], 3).cuda(self.cuda_id) # -pi/2 ~ pi/2
        angles[:,1] = (-math.pi*90/180 - math.pi*90/180) * angles[:,1] + math.pi*90/180  # -pi/2 ~ pi/2
        angles[:,[0,2]] = (-math.pi/12 - math.pi/12) * angles[:,[0,2]] + math.pi/12  # -pi/12 ~ pi/12
        angles[:,1] = torch.clamp(ori_angles[:,1] + angles[:,1], -math.pi/2, math.pi/2)
        angles[:,[0,2]] = torch.clamp(ori_angles[:,[0,2]] + angles[:,[0,2]], -math.pi/6, math.pi/6)

        rotation_m = Compute_rotation_matrix(angles)
        rotated_shape = torch.matmul(front_face, rotation_m.cuda(self.cuda_id))

        # Mesh with color from image
        rot = Meshes(verts=rotated_shape, faces=self.tri[:rot.shape[0],...], textures=Textures(verts_rgb=color_from_img))
        rot = self.phong_renderer(meshes_world=rot, R=self.R, T=self.T)
        rot = torch.clamp(rot, 0.0, 1.0)[...,:3]
        
        # projection matrix
        if _need_const.gpu_p_matrix is None:
            _need_const.gpu_p_matrix = _need_const.p_matrix.cuda(self.cuda_id)
        p_matrix = _need_const.gpu_p_matrix.expand(rotated_shape.shape[0], 3, 3)
        aug_projection = rotated_shape.clone().detach()
        aug_projection[:,:,2] = _need_const.cam_pos - aug_projection[:,:,2]
        aug_projection = aug_projection.bmm(p_matrix.permute(0,2,1))
        face_projection = aug_projection[:,:,0:2] / aug_projection[:,:,2:]
        z_buffer = _need_const.cam_pos - aug_projection[:,:,2:]

        # Re-texturing
        color_from_img = self.get_colors_from_image(rot, face_projection, z_buffer, normalized=True)
        rot = Meshes(verts=face_shape, faces=self.tri[:rot.shape[0],...], textures=Textures(verts_rgb=color_from_img))
        rot = self.phong_renderer(meshes_world=rot, R=self.R, T=self.T)
        rot = torch.clamp(rot[...,:3], 0.0, 1.0)

        occ_mask = self.get_occlusion_mask(parsing_net, scd, rot.permute(0,3,1,2), gui, image[:,[2,1,0],...], obj=False) 
        occ_mask = (occ_mask*3 + mask_only_obj) / 4  # option
        
        # rendered, gui_ori.permute(0,2,3,1)
        return rot, gui.permute(0,2,3,1), occ_mask

    def generate_testing_pairs(self, image, pose=[5.0, 0.0, 0.0], front=False, landmark=False):
    
        # Regress 3DMM parameters
        coef = self.regress_3dmm(image) # RGB2BGR
        # Reconstruct shape and color from 3DMM parameters
        face_shape, angles, _, face_color, _, face_projection, z_buffer, front_face = Reconstruction(coef,self.face_model)
        # get color value from image
        color_from_img = self.get_colors_from_image(image, face_projection, z_buffer, normalized=True)

        if not front:
            # get rotation angles and rotate (x:pitch, y:yaw, z:roll)
            pose = torch.FloatTensor(pose)
            angles = torch.zeros(coef.shape[0], 3)
            angles[:] = math.pi*pose/180
            rotation_m = Compute_rotation_matrix(angles)
            rotated_shape = torch.matmul(front_face, rotation_m.cuda(self.cuda_id))
        else:
            angles[:,[1,2]] = 0.0
            rotation_m = Compute_rotation_matrix(angles)
        rotated_shape = torch.matmul(front_face, rotation_m.cuda(self.cuda_id))

        # Mesh with color from image
        rotated = Meshes(verts=rotated_shape, faces=self.tri[:image.shape[0],...], textures=Textures(verts_rgb=color_from_img))
        rotated = self.phong_renderer(meshes_world=rotated, R=self.R, T=self.T)
        rotated = torch.clamp(rotated, 0.0, 1.0)[...,:3]

        guidance = Meshes(verts=rotated_shape, faces=self.tri[:image.shape[0],...], textures=Textures(verts_rgb=face_color[...,[2,1,0]]))
        guidance = self.phong_renderer(meshes_world=guidance, R=self.R, T=self.T)
        guidance = torch.clamp(guidance, 0.0, 1.0)[...,:3]

        if landmark:
            face_projection, _ = Projection_layer(front_face, rotation_m, rotation_m.new_zeros(angles.shape[0],3))
            landmarks_2d = torch.zeros_like(face_projection).cuda(self.cuda_id)
            landmarks_2d[...,0] = torch.clamp(face_projection[...,0].clone(), 0, self.render_size-1)
            landmarks_2d[...,1] = torch.clamp(face_projection[...,1].clone(), 0, self.render_size-1)
            landmarks_2d[...,1] = self.render_size - landmarks_2d[...,1].clone() - 1
            landmarks_2d = landmarks_2d[:,self.face_model.keypoints,:]

            return rotated, guidance, landmarks_2d

        return rotated, guidance
    
if __name__=='__main__':
    from faceParsing.model import BiSeNet
    # params
    batch_size = 2
    cuda_id = 0
    estimator_path = "saved_models/trained_weights_occ_3d.pth"
    input_img_path = "./test_imgs/input"
    save_path = "./test_imgs/output"
    
    # create models
    # Face parsing network
    bisenet = BiSeNet(n_classes=19)
    bisenet.load_state_dict(torch.load('faceParsing/model_final_diss.pth', map_location='cuda:'+str(cuda_id)))
    bisenet.cuda(cuda_id)
    bisenet.eval()
    # 3D face regressor
    estimator = Estimator3D(batch_size=batch_size, render_size=224, test=True, model_path=estimator_path, cuda_id=cuda_id)
    # structure and contrast difference
    scd = SCDiffer()
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    img_list = glob.glob(input_img_path + '/*.jpg')
    print('The number of images:', len(img_list))
    
    for i in tqdm(range(0, len(img_list), batch_size)):
        until = i+batch_size
        if until > len(img_list):
            until = len(img_list)
        input_img = estimator.align_convert2tensor(img_list[i:until], aligned=True)
        rot, gui, occ_mask = estimator.swap_and_rotate_and_render(input_img, bisenet, scd)
        
        occ_mask = occ_mask.permute(0,2,3,1).cpu().numpy() * 255.0
        gui = gui.cpu().numpy() * 255.0
        rot = rot.cpu().numpy() * 255.0
                        
        for k in range(rot.shape[0]):
            cv2.imwrite(os.path.join(save_path,os.path.basename(img_list[i+k])[:-4]+'_occ.jpg'), occ_mask[k])
            cv2.imwrite(os.path.join(save_path,os.path.basename(img_list[i+k])[:-4]+'_rot.jpg'), cv2.cvtColor(rot[k], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_path,os.path.basename(img_list[i+k])[:-4]+'_gui.jpg'), cv2.cvtColor(gui[k], cv2.COLOR_RGB2BGR))