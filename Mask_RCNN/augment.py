import numpy as np
from random import *

class Agg_Func():
    def __init__(self, funcs=[]):
        self.funcs = funcs
        
    def __call__(self, x, image_shape=None, reverse=False):
        res = x
        for f_ix in range(len(self.funcs)):
            if reverse is True:
                # should do reversely
                # for example: rot90 + lr flip not equals to rot90 (k=-1) + lr flip
                f = self.funcs[len(self.funcs)-1-f_ix]
            else:
                f = self.funcs[f_ix]

            if f == np.rot90 and reverse is True:
                res = f(res, k=-1)
            else:
                if image_shape is not None:
                    res = f(res, image_shape)
                    if isinstance(f, box_rot_back_ang):
                        image_shape[0], image_shape[1] = image_shape[1], image_shape[0]
                        # since rotateted, the image shape is different then
                else:
                    res = f(res)
                    
        return res
    
    
class box_fliplr():
    def __init__(self):
        pass
    
    # coordinate = [y1, x1, y2, x2]
    def __call__(self, coordinate, image_shape):
        #print(image_shape, coordinate)
        y1, x1, y2, x2 = coordinate
        #print('after lr', [y1, image_shape[1] - x2, y2, image_shape[1] - x1])
        return [y1, image_shape[1] - x2, y2, image_shape[1] - x1]
    
class box_flipud():
    def __init__(self):
        pass
    
    # coordinate = [y1, x1, y2, x2]
    def __call__(self, coordinate, image_shape):
        #print(image_shape, coordinate)
        y1, x1, y2, x2 = coordinate
        #print('after ud', [image_shape[0]-y2, x1, image_shape[0]-y1, x2])
        return [image_shape[0]-y2, x1, image_shape[0]-y1, x2]
    
class box_rot_back_ang():
    def __init__(self, degree):
        self.rad = np.deg2rad(-1.*degree)
        #print(self.rad)
        
    def rot(self, xy, image_shape):
        
        org_center = [(image_shape[1]-1) / 2., (image_shape[0]-1) / 2.]
        rot_center = [org_center[1], org_center[0]]
        org = np.array(xy).astype(np.float32) - np.array(org_center)
        
        new = np.array([
                org[0]*np.cos(self.rad)+org[1]*np.sin(self.rad),
                -org[0]*np.sin(self.rad)+org[1]*np.cos(self.rad),
            ])
        #print(org_center, rot_center, org, new, (new+rot_center).astype(np.uint32))

        res = new+rot_center
        res[0] = max(res[0], 0.)
        res[1] = max(res[1], 0.)
        return res.astype(np.uint32)
        
    # coordinate = [y1, x1, y2, x2]
    def __call__(self, coordinate, image_shape):
        
        #print(image_shape, coordinate)
        y1, x1, y2, x2 = coordinate
        x1, y1 = self.rot([x1, y1], image_shape)
        x2, y2 = self.rot([x2, y2], image_shape)
        #print('after rot90', [min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)])
        
        res = [min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)]
        res[0] = max(res[0], 0.)
        res[1] = max(res[1], 0.)
        res[2] = min(res[2], image_shape[1])
        res[3] = min(res[3], image_shape[0])
        return res
    
class Augment():
    def __init__(self, rotate=True, flip=True, image_shape=(100,100)):
        self.augment_funcs = []
        self.box_reverse_augment_funcs = []
        
        if flip:   
            self.augment_funcs.extend([
                Agg_Func(funcs=[]),
                Agg_Func(funcs=[np.fliplr]),
                Agg_Func(funcs=[np.flipud]),
                Agg_Func(funcs=[np.fliplr, np.flipud]),
            ])
            
            self.box_reverse_augment_funcs.extend([
                Agg_Func(funcs=[]),
                Agg_Func(funcs=[box_fliplr()]),
                Agg_Func(funcs=[box_flipud()]),
                Agg_Func(funcs=[box_fliplr(), box_flipud()]),
            ])
            
            if rotate:
                self.augment_funcs.extend([
                    Agg_Func(funcs=[np.rot90]),
                    Agg_Func(funcs=[np.rot90, np.fliplr]),
                    Agg_Func(funcs=[np.rot90, np.flipud]),
                    Agg_Func(funcs=[np.rot90, np.fliplr, np.flipud]),
                ])
                
                self.box_reverse_augment_funcs.extend([
                    Agg_Func(funcs=[box_rot_back_ang(90)]),
                    Agg_Func(funcs=[box_rot_back_ang(90), box_fliplr()]),
                    Agg_Func(funcs=[box_rot_back_ang(90), box_flipud()]),
                    Agg_Func(funcs=[box_rot_back_ang(90), box_fliplr(), box_flipud()]),
                ])
        elif rotate:
            self.augment_funcs.extend([
                Agg_Func(funcs=[]),
                Agg_Func(funcs=[np.rot90]),
            ])
            
            self.box_reverse_augment_funcs.extend([
                Agg_Func(funcs=[]),
                Agg_Func(funcs=[box_rot_back_ang(90)]),
            ])
    
    def get_random_augement(self):
        index = randrange(len(self.augment_funcs))
        return self.augment_funcs[index]
    
    def __len__(self):
        return len(self.augment_funcs)
    
    def apply_image_augment(self, image, index, is_reverse=False):
        f = self.augment_funcs[index]
        return f(image, reverse=is_reverse) 
    
    # boxes = [N, (y1, x1, y2, x2)]
    def apply_boxes_augment(self, boxes, index, image_shape):
        if len(boxes) == 0:
            return []
        
        res = []
        f = self.box_reverse_augment_funcs[index]        
        for b in boxes:
            reversed_box = f(b, reverse=True, image_shape=np.copy(image_shape))
            for shit in reversed_box:
                if shit < 0:
                    print('SHIT!')
                    
            res.append(reversed_box) # f modifies image shape, we shall pass a copy to it      
            
        return np.stack(res, axis=0)
    
    # masks = [height, width, N]
    def apply_masks_augment(self, masks, index, is_reverse=False):
        if len(masks) == 0:
            return []
        
        res = []
        f = self.augment_funcs[index]        
        for i in range(masks.shape[2]):
            m = masks[:,:,i]
            res.append(f(m, reverse=is_reverse))

        return np.stack(res, axis=2)  
        
    
    