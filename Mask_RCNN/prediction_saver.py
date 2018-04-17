import numpy as np
import pandas as pd
import pickle
import gzip
from soft_nms import soft_nms_summarize
from multiprocessing import Pool
import os
from concurrent.futures import ThreadPoolExecutor as thread_pool
from concurrent.futures import ProcessPoolExecutor as process_pool
from concurrent.futures import as_completed
import tqdm
            
def concat(np1, np2):
    if len(np1) == 0:
        return np.array(np2)
    if len(np2) == 0:
        return np.array(np1)
        
    return np.concatenate([np1, np2], axis=0)

def perform_final_nms(packed_data):
    ps, image_id, min_confidence, total_vote_count, ratio, sigma = packed_data
    
    data = ps.load(image_id)

    filtered_boxes, filtered_masks, filtered_scores, filtered_class_ids, nb_filter_masks = \
        soft_nms_summarize(data['boxes'], data['masks'], data['scores'], 
                              np.ones((len(data['boxes']),)), data['sample_counts'], 
                              min_confidence, total_vote_count, ratio=ratio, sigma=sigma)

    nb_data = {
               'boxes': filtered_boxes,
               'masks': nb_filter_masks,
               'scores': filtered_scores,
               'sample_counts': [1],
              }
    ps.save(image_id, nb_data, is_final=True)    
    
    print(image_id, 'done.')
    #return (filtered_boxes, filtered_masks, filtered_scores, filtered_class_ids, nb_filter_masks)
    
def perform_final_nms_with_other_models(packed_data):
    ps, other_ps, image_id, min_confidence, total_vote_count, ratio, sigma = packed_data
    
    data = ps.load(image_id, is_final=True)

    new_data = {
        'boxes': np.array(data['boxes']),
        'scores': np.array(data['scores']),
        'masks': np.array(data['masks'])
    }
    
    for op in other_ps:
        other_data = op.load(image_id, is_final=True)
        new_data['boxes'] = concat(new_data['boxes'], other_data['boxes'])
        new_data['scores'] = concat(new_data['scores'], other_data['scores'])
        new_data['masks'] = concat(new_data['masks'], other_data['masks'])
    
    del data;
    data = new_data
    
    filtered_boxes, filtered_masks, filtered_scores, filtered_class_ids, nb_filter_masks = \
        soft_nms_summarize(data['boxes'], data['masks'], data['scores'], 
                              np.ones((len(data['boxes']),)), np.ones((len(data['boxes']),)), 
                              min_confidence, total_vote_count, ratio=ratio, sigma=sigma, is_intra=False)
    
    nb_data = {
               'boxes': filtered_boxes,
               'masks': filtered_masks,
               'scores': filtered_scores,
               'sample_counts': [1],
              }
    ps.save_other_folder(image_id, nb_data, '/ensemble')   
    
    print(image_id, 'done.')
    
class prediction_saver():
    def __init__(self, path, image_ids):
        self.save_folder_path = path
        self.finalized_folder_path = path + '/final'
        self.image_ids = image_ids
        
        if not os.path.exists(self.save_folder_path):
            os.makedirs(self.save_folder_path)   
            
        if not os.path.exists(self.finalized_folder_path):
            os.makedirs(self.finalized_folder_path)
            
    def save(self, image_id, data, is_final=False):
        
        if is_final is True:
            final_path = self.finalized_folder_path + '/' + str(image_id) + '.npz'
        else:
            final_path = self.save_folder_path + '/' + str(image_id) + '.npz'
        
        np.savez_compressed(final_path, 
                            masks=np.array(data['masks']), 
                            boxes=np.array(data['boxes']), 
                            scores=np.array(data['scores']), 
                            sample_counts=np.array(data['sample_counts']))
        
        '''
        fp=gzip.open(final_path,'wb')
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        fp.close()
        
        with open(final_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        '''
    
    def save_other_folder(self, image_id, data, other_path):
        
        other_folder = self.save_folder_path + other_path
        if not os.path.exists(other_folder):
            os.makedirs(other_folder)   
            
        final_path = other_folder + '/' + str(image_id) + '.npz'
        np.savez_compressed(final_path, 
                            masks=np.array(data['masks']), 
                            boxes=np.array(data['boxes']), 
                            scores=np.array(data['scores']), 
                            sample_counts=np.array(data['sample_counts']))
        
    def load_other_folder(self, image_id, other_path):  
        
        final_folder = self.save_folder_path + other_path
        final_path = final_folder + '/' + str(image_id) + '.npz'
        
        if (str(image_id) + '.npz') not in os.listdir(final_folder):
            return {
                'masks': np.array([]),
                'boxes': np.array([]),
                'scores': np.array([]),
                'sample_counts': np.array([])
                }
            
        
        data = np.load(final_path)
        return data
        
    def load(self, image_id, is_final=False):  
        
        if is_final is True:
            final_folder = self.finalized_folder_path
            final_path = self.finalized_folder_path + '/' + str(image_id) + '.npz'
        else:
            final_folder = self.save_folder_path
            final_path = self.save_folder_path + '/' + str(image_id) + '.npz'
        
        if (str(image_id) + '.npz') not in os.listdir(final_folder):
            return {
                'masks': np.array([]),
                'boxes': np.array([]),
                'scores': np.array([]),
                'sample_counts': np.array([])
                }
            
        
        data = np.load(final_path)
        
        '''
        fp=gzip.open(final_path,'rb') # This assumes that primes.data is already packed with gzip
        data=pickle.load(fp)
        fp.close()
        '''
        
        return data
    
    def update_prediction(self, image_id, masks, boxes, scores, sample_counts):
        
        #data = self.load(image_id)
        
        data = {
                'masks': np.array([]),
                'boxes': np.array([]),
                'scores': np.array([]),
                'sample_counts': np.array([])
                }
        data['masks'] = concat(data['masks'], masks)
        data['boxes'] = concat(data['boxes'], boxes)
        data['scores'] = concat(data['scores'], scores)
        data['sample_counts'] = concat(data['sample_counts'], sample_counts)
            
        self.save(image_id, data)

    
    def merge_data(self, other_prediction_saver):
        other_data= other_prediction_saver.get_data()
        
        for image_id in self.image_ids:
            other_data = other_prediction_saver.load(image_id)
            
            self.update_prediction(image_id, other_data['masks'], other_data['boxes'],
                                  other_data['scores'], other_data['sample_counts'])

    def finalize_predictions(self, image_ids, min_confidence, total_vote_count, ratio, sigma):
        
        # ps, image_id, min_confidence, total_vote_count, ratio, sigma = packed_data
        final_res = []
        arguments = [(self, image_id, min_confidence, total_vote_count, ratio, sigma) for image_id in image_ids]
        print(len(image_ids))
        with thread_pool(max_workers=6) as executor:
            futures = [executor.submit(perform_final_nms, arg) for arg in arguments]
            
            for i, future in enumerate(as_completed(futures)):
                res = future.result()
                print('Finshed count=', i+1, '\n')
                
            '''
            for i, res in enumerate(executor.map(perform_final_nms, arguments)):
                # (filtered_boxes, filtered_masks, filtered_scores, filtered_class_ids, nb_filter_masks)
                nb_data = {
                   'boxes': res[0],
                   'masks': res[4],
                   'scores': res[1],
                   'sample_counts': [1],
                }
                self.save(image_ids[i], nb_data, is_final=True)
                final_res.append(res)
                #print(i)
                
            ''' 
        # return final_res
        
        '''        
        p = Pool(6)
        res = p.imap(perform_final_nms, 
                    [(self, image_id, min_confidence, total_vote_count, ratio, sigma) for image_id in image_ids],
                     chunksize=len(image_ids)//6)
        res = list(res)
        print('Done processing, Start Saving files')        
        for i, image_id in enumerate(image_ids):
            nb_data = {
               'boxes': res[i][0],
               'masks': res[i][4],
               'scores': res[i][2],
               'sample_counts': [],
            }
            self.save(image_id, nb_data, is_final=True)
        p.terminate()
        p.join()  
        '''
        
    
    # other_ps: [ps1, ps2, ps3, ...]
    def ensemble_other_predictions(self, other_ps, image_ids, min_confidence, total_vote_count, ratio, sigma):
        
        final_res = []
        arguments = [(self, other_ps, image_id, min_confidence, total_vote_count, ratio, sigma) for image_id in image_ids]
        print(len(image_ids))
        with thread_pool(max_workers=1) as executor:
            futures = [executor.submit(perform_final_nms_with_other_models, arg) for arg in arguments]
            
            for i, future in enumerate(as_completed(futures)):
                res = future.result()
                print('Finshed count=', i+1, '\n')
                
        '''        
        p = Pool(6)
        res = p.imap(perform_final_nms_with_other_models, 
                    [(self, other_ps, image_id, min_confidence, total_vote_count, ratio, sigma) for image_id in image_ids],
                    chunksize=len(image_ids)//6)
        
        res = list(res)
        print('Done processing')
        p.terminate()
        p.join()
        
        return res
        '''