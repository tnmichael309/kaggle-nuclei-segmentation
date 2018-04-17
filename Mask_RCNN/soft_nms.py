# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# ----------------------------------------------------------

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.filters import threshold_otsu as th_func

# brief: check the overlaps of two masks
def calc_box_overlap_ratio(box1, box2):
    
    ty1 = box1[0]
    tx1 = box1[1]
    ty2 = box1[2]
    tx2 = box1[3]
       
    y1 = box2[0]
    x1 = box2[1]
    y2 = box2[2]
    x2 = box2[3]

    area1 = (tx2 - tx1 + 1) * (ty2 - ty1 + 1)
    area2 = (x2 - x1 + 1) * (y2 - y1 + 1)
    iw = (min(tx2, x2) - max(tx1, x1) + 1)
    ih = (min(ty2, y2) - max(ty1, y1) + 1)
    
    if iw > 0 and ih > 0:
        ua = float(area1 + area2 - iw*ih)
        return iw * ih / ua
    else:
        return 0.

def calc_mask_overlap_ratio(mask1, mask2):
    bin_mask1 = np.where(mask1 >= 0.5, 1, 0).astype(np.uint8)
    bin_mask2 = np.where(mask2 >= 0.5, 1, 0).astype(np.uint8)
    
    return np.logical_and(bin_mask1, bin_mask2).sum() * 1. / min(bin_mask1.sum(), bin_mask2.sum()) #np.logical_or(bin_mask1, bin_mask2).sum()
    
def merge_into_large_box(box1, box2):
    ty1 = box1[0]
    tx1 = box1[1]
    ty2 = box1[2]
    tx2 = box1[3]
       
    y1 = box2[0]
    x1 = box2[1]
    y2 = box2[2]
    x2 = box2[3]
    
    return [min(ty1, y1), min(tx1, x1), max(ty2, y2), max(tx2, x2)]

# boxes = [similar_count, (y1, x1, y2, x2)]
# masks = [similar_count, image height, image width]
def merge_masks_and_boxes_and_scores(masks, boxes, scores, sample_nums, is_intra=True):
    
    old_scores = scores
    scores = np.array([scores[i]*sample_nums[i] for i, s in enumerate(scores)]) # readjust scores accoding to sample nums
    scores = scores / scores.sum() # normalize scores
    
    if is_intra is True:
        # calculate weight matrix for each mask
        score_matrix = np.zeros(masks.shape)
        for i in range(len(scores)):
            score_matrix[i,:,:] = np.where(masks[i,:,:] > 0.1, scores[i], 0) # restore orignal score sum

        # normalize score matrix
        eps = 1e-6
        score_sum = score_matrix[:,:,:].sum(axis=0) + eps # to avoid warning
        for i in range(len(scores)):
            score_matrix[i,:,:] = np.where(score_sum>eps, score_matrix[i,:,:]/score_sum, 0)
            
        final_mask = (masks*score_matrix).sum(axis=0)
    else:
        final_mask = np.sum([masks[i,:,:]*scores[i] for i in range(len(scores))], axis=0)
    
    final_box = np.array([min(boxes[:, 0]), min(boxes[:, 1]), max(boxes[:, 2]), max(boxes[:, 3])])
    #final_box = np.sum([boxes[i]*scores[i] for i in range(boxes.shape[0])], axis = 0).astype(np.uint32)

    return final_mask, final_box, sum([old_scores[i]*sample_nums[i] for i, s in enumerate(old_scores)])/sum(sample_nums)

def flexible_avg_with_precomputed_overlap_ratios(boxes, masks, scores, class_ids, overlap_ratios, top_N=None, 
                                                 sample_counts=None, is_intra=True):
    N = len(boxes)
    if N == 0:
        return [], [], [], [], []
    
    if sample_counts is None:
        sample_counts = np.ones((N,))
    
    filtered_boxes = []
    filtered_masks = []
    filtered_scores = []
    filtered_class_ids = []
    filtered_sample_nums = []
    is_merged = [False] * N
    
    for i in range(N):   
        if is_merged[i] is True:
            continue
            
        current_box = boxes[i]
        current_mask = masks[:,:,i]
        current_score = scores[i]
        current_sample_num = sample_counts[i]
        
        overlapped_masks = [current_mask]
        overlapped_boxes = [current_box]
        overlapped_scores = [current_score]
        overlapped_sample_nums = [current_sample_num]
        
        for j in range(i+1, N):
            if is_merged[j] is True:
                continue
            
            check_box = boxes[j]
            check_mask = masks[:,:,j]
            check_score = scores[j]
            check_sample_num = sample_counts[j]
            
            # do not use calc_box_overlap_ratio(current_box, check_box) => # wrong case: long and inclined cell
            overlap_ratio = overlap_ratios[i, j] # calc_mask_overlap_ratio(current_mask, check_mask)
            
            if overlap_ratio > .8:
                is_merged[j] = True
                
                if top_N is None or len(overlapped_masks) < top_N:
                    overlapped_masks.append(check_mask)
                    overlapped_boxes.append(check_box)
                    overlapped_scores.append(check_score)
                    overlapped_sample_nums.append(check_sample_num)
                    
        if len(overlapped_masks) > 1:
            current_mask, current_box, current_score = merge_masks_and_boxes_and_scores(np.array(overlapped_masks), 
                                                                                        np.array(overlapped_boxes), 
                                                                                        np.array(overlapped_scores),
                                                                                        np.array(overlapped_sample_nums),
                                                                                        is_intra=is_intra)
        
        # if total_vote_count is specified, means we have group#=total_vote_count trying to combine
        # we have to at least half of the members say it exist, then it exist
        filtered_boxes.append(current_box)
        filtered_masks.append(current_mask)
        filtered_scores.append(current_score)
        filtered_class_ids.append(class_ids[i])
        filtered_sample_nums.append(sum(overlapped_sample_nums))
        
    return filtered_boxes, filtered_masks, filtered_scores, filtered_class_ids, filtered_sample_nums

def flexible_avg(boxes, masks, scores, class_ids, top_N=None, sample_counts=None):
    
    N = len(boxes)
    if N == 0:
        return [], [], [], [], []
    
    if sample_counts is None:
        sample_counts = np.ones((N,))

    filtered_boxes = []
    filtered_masks = []
    filtered_scores = []
    filtered_class_ids = []
    filtered_sample_nums = []
    is_merged = [False] * N
    
    for i in tqdm(range(N)):   
        if is_merged[i] is True:
            continue
            
        current_box = boxes[i]
        current_mask = masks[:,:,i]
        current_score = scores[i]
        current_sample_num = sample_counts[i]
        
        overlapped_masks = [current_mask]
        overlapped_boxes = [current_box]
        overlapped_scores = [current_score]
        overlapped_sample_nums = [current_sample_num]
        
        for j in range(i+1, N):
            if is_merged[j] is True:
                continue
            
            check_box = boxes[j]
            check_mask = masks[:,:,j]
            check_score = scores[j]
            check_sample_num = sample_counts[j]
            
            # do not use calc_box_overlap_ratio(current_box, check_box) => # wrong case: long and inclined cell
            overlap_ratio = calc_mask_overlap_ratio(current_mask, check_mask)
            if overlap_ratio > .8:
                is_merged[j] = True
                
                if top_N is None or len(overlapped_masks) < top_N:
                    overlapped_masks.append(check_mask)
                    overlapped_boxes.append(check_box)
                    overlapped_scores.append(check_score)
                    overlapped_sample_nums.append(check_sample_num)
                    
        if len(overlapped_masks) > 1:
            current_mask, current_box, current_score = merge_masks_and_boxes_and_scores(np.array(overlapped_masks), 
                                                                                        np.array(overlapped_boxes), 
                                                                                        np.array(overlapped_scores),
                                                                                        np.array(overlapped_sample_nums))
        
        # if total_vote_count is specified, means we have group#=total_vote_count trying to combine
        # we have to at least half of the members say it exist, then it exist
        filtered_boxes.append(current_box)
        filtered_masks.append(current_mask)
        filtered_scores.append(current_score)
        filtered_class_ids.append(class_ids[i])
        filtered_sample_nums.append(sum(overlapped_sample_nums))
        
    return filtered_boxes, filtered_masks, filtered_scores, filtered_class_ids, filtered_sample_nums

def parallel_flexible_avg(data):
    rois, masks, scores, class_ids = data

    rois, masks, scores, class_ids, sample_nums = \
        flexible_avg(rois, masks, scores, class_ids) # flexibly average all with iou > .8

    return [rois, masks, scores, class_ids, sample_nums]
     
def soft_nms_summarize(boxes_arr, masks_arr, scores_arr, class_ids_arr, sample_nums_arr, min_confidence, 
                       total_vote_count, ratio=.5, sigma=0.5, is_intra=True):
    
    N = len(boxes_arr)
    if N == 0:
        print('No masks')
        return [], [], [], [], []
    
    # print('Summarizing')
    # gather boxes, masks and scores
    boxes = np.array(boxes_arr)
    masks = np.array(masks_arr)
    scores = np.array(scores_arr)
    class_ids = np.array(class_ids_arr)
    sample_nums = np.array(sample_nums_arr)
    
    # sort by scores
    index = np.argsort(scores)
    boxes = boxes[index]
    masks = masks[index]
    scores = scores[index]
    class_ids = class_ids[index]
    sample_nums = sample_nums[index]
    
    
    # check masks to remain
    # print('Filtering remaining masks')
    is_remain = [False]*len(sample_nums)
    total_size = len(sample_nums)
    overlap_ratios = np.zeros((total_size,total_size))
    
    for i in range(total_size):
        
        overlap_count = 0
        for j in range(total_size):
            
            if j == i:
                overlap_ratio = 1.
            elif j > i:
                overlap_ratio = calc_mask_overlap_ratio(masks[i], masks[j])
                overlap_ratios[i,j] = overlap_ratio
                overlap_ratios[j,i] = overlap_ratio
            else:
                overlap_ratio = overlap_ratios[i,j]
                
            #print(i, j, overlap_ratio)
            # those surrond it a bit should count
            if overlap_ratio > .5:
                overlap_count += 1
                
                if overlap_count > int(total_vote_count*ratio):
                    is_remain[i] = True
                
    original_masks_num = len(sample_nums)
    boxes = boxes[is_remain]
    masks = masks[is_remain]
    scores = scores[is_remain]
    class_ids = class_ids[is_remain]
    sample_nums = sample_nums[is_remain]
    overlap_ratios = overlap_ratios[is_remain,:]
    overlap_ratios = overlap_ratios[:, is_remain]
    
    print('Orignal masks# =', original_masks_num, ' => Voted for next level flexible nms masks# =', len(sample_nums))
    
    
    # to follow cpu_flexible_nms required format 
    masks = np.transpose(masks, axes=(1,2,0))

    filtered_boxes, filtered_masks, filtered_scores, filtered_class_ids, _ = \
        flexible_avg_with_precomputed_overlap_ratios(boxes, masks, scores, class_ids, overlap_ratios, 
                                                     top_N=None, sample_counts=sample_nums, is_intra=is_intra)
        #flexible_avg(boxes, masks, scores, class_ids, top_N=None, sample_counts=sample_nums)
        
    return mask_soft_nms(filtered_boxes, filtered_masks, filtered_scores, filtered_class_ids, min_confidence, sigma=sigma)


def mask_soft_nms(filtered_boxes, filtered_masks, filtered_scores, filtered_class_ids, min_confidence, sigma=0.5):
    
    # dirty soft nms part
    filtered_boxes = np.array(filtered_boxes)
    filtered_masks = np.array(filtered_masks)
    filtered_scores = np.array(filtered_scores)
    filtered_class_ids = np.array(filtered_class_ids)
    
    N = len(filtered_boxes)
    if N == 0:
        return filtered_boxes, filtered_masks, filtered_scores, filtered_class_ids, np.copy(filtered_masks)
    
    for i in range(N):   
        maxpos = np.argmax(filtered_scores[i:])
        maxpos += i # the offset!!!!
        
        filtered_boxes[[i,maxpos]] = filtered_boxes[[maxpos,i]] 
        filtered_masks[[i,maxpos]] = filtered_masks[[maxpos,i]] 
        filtered_scores[[i,maxpos]] = filtered_scores[[maxpos,i]] 
        filtered_class_ids[[i,maxpos]] = filtered_class_ids[[maxpos,i]] 
        #overlap_ratios[[i,maxpos],:] = overlap_ratios[[maxpos,i],:]
        #overlap_ratios[:,[i,maxpos]] = overlap_ratios[:,[maxpos,i]]
        
        current_box = filtered_boxes[i]
        current_mask = filtered_masks[i]
        current_score = filtered_scores[i]
        
        for j in range(i+1, N):
            check_box = filtered_boxes[j]
            check_mask = filtered_masks[j]
            check_score = filtered_scores[j]
            
            #overlap_ratio = overlap_ratios[i, j] 
            overlap_ratio = calc_mask_overlap_ratio(current_mask, check_mask)
            #overlap_ratio = calc_box_overlap_ratio(current_box, check_box)
            #print(current_box, check_box, overlap_ratio)
            
            filtered_scores[j] = filtered_scores[j]*np.exp(-1.*overlap_ratio*overlap_ratio/sigma)
            
    keep_index = filtered_scores > min_confidence
    filtered_boxes = filtered_boxes[keep_index]
    filtered_masks = filtered_masks[keep_index]
    filtered_scores = filtered_scores[keep_index]
    filtered_class_ids = filtered_class_ids[keep_index]
    
    # binarize all masks
    nb_filtered_masks = np.copy(filtered_masks)
    
    for i in range(len(filtered_masks)):
        filtered_masks[i] = np.where(filtered_masks[i]>=0.5, 1, 0).astype(np.uint8)
    
    filtered_masks = np.stack(filtered_masks, axis=2)
    
    return filtered_boxes, filtered_masks, filtered_scores, filtered_class_ids, nb_filtered_masks

# boxes: [N, (y1, x1, y2, x2)]
# masks: [image height, image width, num_instances]  
# top_N = None: Merge all the masks with iou > .8, otherwise keep #=top_N and iou > .8
def cpu_flexible_nms(boxes, masks, scores, class_ids, min_confidence, sigma=0.5, top_N=None, sample_nums=None):
    
    filtered_boxes, filtered_masks, filtered_scores, filtered_class_ids, _ = \
        flexible_avg(boxes, masks, scores, class_ids, top_N=top_N, sample_counts=sample_nums)
        
    return mask_soft_nms(filtered_boxes, filtered_masks, filtered_scores, filtered_class_ids, min_confidence, sigma=sigma)

# used to check all masks above detection min confidence
def cpu_dummy_nms(iou_weights, scores, proposal_count, Nt, min_confidence):
    N = iou_weights.shape[0]
    return np.argsort(scores[:N], kind='mergesort')[:proposal_count].astype(np.int32)
    
def cpu_soft_nms(iou_weights, scores, proposal_count, Nt, min_confidence):
    # iou_weights shape: NxN <= iou overlaps of every possible pairs in boxes
    
    #print(boxes, '\n', scores, '\n', proposal_count, '\n', Nt, '\n', sigma, '\n', threshold, '\n', method)
    '''
    In-place modification or storing func input or return values in python datastructures without explicit (np.)copy can have non
    deterministic consequences (See https://www.tensorflow.org/api_docs/python/tf/py_func)
    '''
    
    threshold = min_confidence # min_confidence
    iou_weights = np.copy(iou_weights)
    scores = np.copy(scores)
    N = iou_weights.shape[0]
    original_index = np.array([i for i in range(N)])

    for i in range(N):
    
        # get max scored box pos
        maxpos = np.argmax(scores[i:])
        maxpos += i # the offset!!!!
        
        # swap ith box with position of max box
        # need to exchange rows and columns
        iou_weights[[i, maxpos],:] = iou_weights[[maxpos, i],:]
        iou_weights[:,[i, maxpos]] = iou_weights[:,[maxpos, i]]
         
        scores[[i, maxpos]] = scores[[maxpos, i]]
        original_index[[i, maxpos]] = original_index[[maxpos, i]]
        
        scores[i+1:] = scores[i+1:]*iou_weights[i,i+1:]
        #scores = np.where(np.arange(0,N) <= i, scores, scores*iou_weights[i, :])

        
    # since scores only get smaller and smaller, we don't need to sort again
    try:
        filt = scores >= min_confidence
    except:
        print('Exception:\bScores:\n', scores, 'min_confidence:\n', min_confidence)
        
    ret_index = original_index[filt]
    
    return ret_index[:proposal_count].astype(np.int32)

'''
# old version

# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# ----------------------------------------------------------

import numpy as np

def cpu_soft_nms(boxes, scores, proposal_count, Nt, method=2, sigma=0.5):
    #print(boxes, '\n', scores, '\n', proposal_count, '\n', Nt, '\n', sigma, '\n', threshold, '\n', method)
  
    
    boxes = np.copy(boxes)
    scores = np.copy(scores)
    N = boxes.shape[0]
    
    for i in range(N):
    
        # get max box
        maxpos = np.argmax(scores[i:])
        maxscore = scores[maxpos]
        
        # swap ith box with position of max box
        temp = boxes[maxpos,:]
        boxes[[i, maxpos]] = boxes[[maxpos, i]]
        scores[[i, maxpos]] = scores[[maxpos, i]]
        
        ty1 = boxes[i,0]
        tx1 = boxes[i,1]
        ty2 = boxes[i,2]
        tx2 = boxes[i,3]
        ts = scores[i]

        # to calulate overlaps between boxes[i] and boxes[pos], where pos is in [i+1, N)
        def overlap_func(pos):
            y1 = boxes[pos, 0]
            x1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = scores[pos]
            
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            ih = (min(ty2, y2) - max(ty1, y1) + 1)
            ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
            ov = iw * ih / ua #iou between max box and detection box
             
            if method == 1: # linear
                if ov > Nt: 
                    weight = 1 - ov
                else:
                    weight = 1
            elif method == 2: # gaussian
                weight = np.exp(-(ov * ov)/sigma)
            else: # original NMS
                if ov > Nt: 
                    weight = 0
                else:
                    weight = 1
            
            scores[pos] = weight*scores[pos]
            
        map(overlap_func, np.arange(i+1, N, 1))

    return np.argsort(scores[:N], kind='mergesort')[:proposal_count].astype(np.int32)
'''