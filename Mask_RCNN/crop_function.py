import numpy as np

def get_cropped_info(img, min_w_overlap, min_h_overlap, crop_size=512, do_horizontal_crop=False, do_vertical_crop=False):
    img_hs = []
    img_vs = []
    
    if do_horizontal_crop is True:
        img_width = img.shape[1]
        cur_h_pos = 0
        
        while True:
            if cur_h_pos + crop_size <= img_width:
                img_hs.append((cur_h_pos, cur_h_pos+crop_size))
                
                if cur_h_pos + crop_size == img_width:
                    break
                    
                cur_h_pos += crop_size - min_w_overlap    
            else:
                img_hs.append((img_width-crop_size, img_width))
                break
                
    if do_vertical_crop is True:
        img_height = img.shape[0]
        cur_v_pos = 0
        
        while True:
            if cur_v_pos + crop_size <= img_height:
                img_vs.append((cur_v_pos, cur_v_pos+crop_size))
                
                if cur_v_pos + crop_size == img_height:
                    break
                    
                cur_v_pos += crop_size - min_h_overlap    
            else:
                img_vs.append((img_height-crop_size, img_height))
                break
    
    new_image_data = []    
    if do_horizontal_crop is True and do_vertical_crop is True:
        for i, hs in enumerate(img_hs):
            for j, vs in enumerate(img_vs):
                new_image_data.append({
                    'img': img[vs[0]:vs[1], hs[0]:hs[1]],
                    'offset': (vs[0], hs[0]),
                    'to_delete': (i>0, i!=len(img_hs)-1, j>0, j!=len(img_vs)-1), # left, right, top, down
                })
                
    elif do_horizontal_crop is True:
        for i, hs in enumerate(img_hs):
            new_image_data.append({
                'img': img[:, hs[0]:hs[1]],
                'offset': (0, hs[0]),
                'to_delete': (i>0, i!=len(img_hs)-1, False, False), # left, right, top, down
            })
            
    elif do_vertical_crop is True:
        for j, vs in enumerate(img_vs):
            new_image_data.append({
                'img': img[vs[0]:vs[1], :],
                'offset': (vs[0], 0),
                'to_delete': (False, False, j>0, j!=len(img_vs)-1), # left, right, top, down
            })    
    else:
        print('Why call this function!?')
        
    return new_image_data
    
# to_delete_info=(left, right, top, down)    
def get_remain_index(masks, to_delete_info=(False, False, False, False)):
    # since the change of iou calculation, the box info is not accurate
    is_remain = np.array([True]*masks.shape[2])

    for i in range(masks.shape[2]):
        if to_delete_info[0] is True and np.any(masks[:,0,i] >= 0.5):
            is_remain[i] = False
            continue
        if to_delete_info[1] is True and np.any(masks[:,-1,i] >= 0.5):
            is_remain[i] = False
            continue
        if to_delete_info[2] is True and np.any(masks[0,:,i] >= 0.5):
            is_remain[i] = False
            continue
        if to_delete_info[3] is True and np.any(masks[-1,:,i] >= 0.5):
            is_remain[i] = False
            continue    
    
    return is_remain
    
if __name__ == '__main__':
    test_img = np.ones((1388, 1056))
    print('\n\nCrop both')
    print(get_cropped_info(test_img, 50, 30, do_horizontal_crop=True, do_vertical_crop=True))
    print('\n\nCrop horizontally only')
    print(get_cropped_info(test_img, 50, 30, do_horizontal_crop=True, do_vertical_crop=False))
    print('\n\nCrop vertically only')
    print(get_cropped_info(test_img, 50, 30, do_horizontal_crop=False, do_vertical_crop=True))