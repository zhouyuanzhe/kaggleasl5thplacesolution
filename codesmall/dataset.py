import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import bisect
import random
import scipy
from scipy.interpolate import interp1d

LHAND = np.arange(468, 489).tolist() # 21
RHAND = np.arange(522, 543).tolist() # 21
POSE  = np.arange(489, 522).tolist() # 33
FACE  = np.arange(0,468).tolist()    #468

REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
][::2]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
][::2]
NOSE=[
    1,2,98,327
]
SLIP = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415,
]
SPOSE = (np.array([
    11,13,15,12,14,16,23,24,
])+489).tolist()

BODY = REYE+LEYE+NOSE+SLIP+SPOSE

def get_indexs(L):
    return sorted([i + j * len(L) for i in range(len(L)) for j in range(len(L)) if i>j])

DIST_INDEX = get_indexs(RHAND)

LIP_DIST_INDEX = get_indexs(SLIP)

POSE_DIST_INDEX = get_indexs(SPOSE)

EYE_DIST_INDEX = get_indexs(REYE)

NOSE_DIST_INDEX = get_indexs(NOSE)


HAND_START = [0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,0,5,9,13,0]
HAND_END = [1,2,3,4,6,7,8,10,11,12,14,15,16,18,19,20,5,9,13,17,17]


point_dim = len(LHAND+RHAND+REYE+LEYE+NOSE+SLIP+SPOSE)*2+len(LHAND+RHAND)*2+len(RHAND)+len(POSE_DIST_INDEX)+len(DIST_INDEX)*2 +len(EYE_DIST_INDEX)*2+len(LIP_DIST_INDEX)

def do_hflip_hand(lhand, rhand):
    rhand[...,0] *= -1
    lhand[...,0] *= -1
    rhand, lhand = lhand,rhand
    return lhand, rhand

def do_hflip_spose(spose):
    spose[...,0] *= -1
    spose = spose[:,[3,4,5,0,1,2,7,6]]
    return spose


def do_hflip_eye(reye,leye):
    reye[...,0] *= -1
    leye[...,0] *= -1
    reye, leye = leye,reye
    return reye, leye

def do_hflip_slip(slip):
    slip[...,0] *= -1
    slip = slip[:,[10,9,8,7,6,5,4,3,2,1,0]+[19,18,17,16,15,14,13,12,11]]
    return slip

def do_hflip_nose(nose):
    nose[...,0] *= -1
    nose = nose[:,[0,1,3,2]]
    return nose

def pre_process(xyz,aug):

    # select the lip, right/left hand, right/left eye, pose, nose parts.
    lip   = xyz[:, SLIP]#20
    lhand = xyz[:, LHAND]#21
    rhand = xyz[:, RHAND]#21
    pose = xyz[:, SPOSE]#8
    reye = xyz[:, REYE]#16
    leye = xyz[:, LEYE]#16
    nose = xyz[:, NOSE]#4
    
    if aug and random.random()>0.7:
        lhand, rhand = do_hflip_hand(lhand, rhand)
        pose = do_hflip_spose(pose)
        reye,leye = do_hflip_eye(reye,leye)
        lip = do_hflip_slip(lip)
        nose = do_hflip_nose(nose)

    xyz = torch.cat([ #(none, 106, 2)
        lhand,
        rhand,
        lip,
        pose,
        reye,
        leye,
        nose,
    ],1)


    # concatenate the frame delta information
    x = torch.cat([xyz[1:,:len(LHAND+RHAND),:]-xyz[:-1,:len(LHAND+RHAND),:],torch.zeros((1,len(LHAND+RHAND),2))],0)
    
    
    # TODO
    ld = lhand[:,:,:2].reshape(-1,len(LHAND),1,2)-lhand[:,:,:2].reshape(-1,1,len(LHAND),2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.reshape(-1,len(LHAND)*len(LHAND))[:,DIST_INDEX]
    
    rd = rhand[:,:,:2].reshape(-1,len(LHAND),1,2)-rhand[:,:,:2].reshape(-1,1,len(LHAND),2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.reshape(-1,len(LHAND)*len(LHAND))[:,DIST_INDEX]
    
    lipd = lip[:,:,:2].reshape(-1,len(SLIP),1,2)-lip[:,:,:2].reshape(-1,1,len(SLIP),2)
    lipd = torch.sqrt((lipd**2).sum(-1))
    lipd = lipd.reshape(-1,len(SLIP)*len(SLIP))[:,LIP_DIST_INDEX]
    
    posed = pose[:,:,:2].reshape(-1,len(SPOSE),1,2)-pose[:,:,:2].reshape(-1,1,len(SPOSE),2)
    posed = torch.sqrt((posed**2).sum(-1))
    posed = posed.reshape(-1,len(SPOSE)*len(SPOSE))[:,POSE_DIST_INDEX]
    
    reyed = reye[:,:,:2].reshape(-1,len(REYE),1,2)-reye[:,:,:2].reshape(-1,1,len(REYE),2)
    reyed = torch.sqrt((reyed**2).sum(-1))
    reyed = reyed.reshape(-1,len(REYE)*len(REYE))[:,EYE_DIST_INDEX]
    
    leyed = leye[:,:,:2].reshape(-1,len(LEYE),1,2)-leye[:,:,:2].reshape(-1,1,len(LEYE),2)
    leyed = torch.sqrt((leyed**2).sum(-1))
    leyed = leyed.reshape(-1,len(LEYE)*len(LEYE))[:,EYE_DIST_INDEX]

    dist_hand=torch.sqrt(((lhand-rhand)**2).sum(-1))

    xyz = torch.cat([xyz.reshape(-1,(len(LHAND+RHAND+REYE+LEYE+NOSE+SLIP+SPOSE))*2), 
                         x.reshape(-1,(len(LHAND+RHAND))*2),
                         ld,
                         rd,
                         lipd,
                         posed,
                         reyed,
                         leyed,
                         dist_hand,
                        ],1)
    
    # fill the nan value with 0
    xyz[torch.isnan(xyz)] = 0
    
    
    
    return xyz

def do_random_affine(xyz,
    scale  = (0.8,1.5),
    shift  = (-0.1,0.1),
    degree = (-15,15),
    p=0.5
):
    # random scale, shufle, degree augmentation
    if np.random.rand()<p:
        if scale is not None:
            scale_ = np.random.uniform(*scale)
            xyz[:,:,0] = scale_*xyz[:,:,0]
            scale_ = np.random.uniform(*scale)
            xyz[:,:,1] = scale_*xyz[:,:,1]

            scale_ = np.random.uniform(*scale)
            xyz[:,LHAND,0] = scale_*xyz[:,LHAND,0]
            scale_ = np.random.uniform(*scale)
            xyz[:,LHAND,1] = scale_*xyz[:,LHAND,1]

            scale_ = np.random.uniform(*scale)
            xyz[:,RHAND,0] = scale_*xyz[:,RHAND,0]
            scale_ = np.random.uniform(*scale)
            xyz[:,RHAND,1] = scale_*xyz[:,RHAND,1]

        if shift is not None:
            shift_ = np.random.uniform(*shift)
            xyz[:,:,0] = xyz[:,:,0] + shift_
            shift_ = np.random.uniform(*shift)
            xyz[:,:,1] = xyz[:,:,1] + shift_

            shift_ = np.random.uniform(*shift)
            xyz[:,LHAND,0] = xyz[:,LHAND,0] + shift_/2
            shift_ = np.random.uniform(*shift)
            xyz[:,LHAND,1] = xyz[:,LHAND,1] + shift_/2

            shift_ = np.random.uniform(*shift)
            xyz[:,RHAND,0] = xyz[:,RHAND,0] + shift_/2
            shift_ = np.random.uniform(*shift)
            xyz[:,RHAND,1] = xyz[:,RHAND,1] + shift_/2

        if degree is not None:
            degree_ = np.random.uniform(*degree)
            radian = degree_/180*np.pi
            c = np.cos(radian)
            s = np.sin(radian)
            rotate = np.array([
                [c,-s],
                [s, c],
            ]).T
            xyz[:,:,:2] = xyz[:,:,:2] @rotate

            degree_ = np.random.uniform(*degree)
            radian = degree_/180*np.pi
            c = np.cos(radian)
            s = np.sin(radian)
            rotate = np.array([
                [c,-s],
                [s, c],
            ]).T
            xyz[:,RHAND,:2] = xyz[:,RHAND,:2] @rotate

            degree_ = np.random.uniform(*degree)
            radian = degree_/180*np.pi
            c = np.cos(radian)
            s = np.sin(radian)
            rotate = np.array([
                [c,-s],
                [s, c],
            ]).T
            xyz[:,LHAND,:2] = xyz[:,LHAND,:2] @rotate

    return xyz
#-----------------------------------------------------
def train_augment(xyz):
    xyz = do_random_affine(
        xyz,
        scale  = (0.8,1.2),
        shift  = (-0.2,0.2),
        degree = (-5,5),
        p=0.7
    )
    return xyz


def do_normalise_by_ref(xyz, ref):  
    K = xyz.shape[-1]
    xyz_flat = ref.reshape(-1,K)
    m = np.nanmean(xyz_flat,0).reshape(1,1,K)
    s = np.nanstd(xyz_flat, 0).mean() 
    xyz = xyz - m
    xyz = xyz / s
    return xyz

class D(Dataset):

    def __init__(self, path, training=False):

        self.data = np.load(path, allow_pickle=True)
        self.maxlen = 256 # 537 actually
        self.training = training
        self.label_map = [[] for _ in range(250)]
        for i, item in enumerate(self.data):
            label = item['label']
            self.label_map[label].append(i)
        if training:
            self.augment = train_augment
        else:
            self.augment = None


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, rec=False):

        tmp = self.data[idx]
        data = tmp['data']
        label = tmp['label']    

        xyz = data.reshape((-1, 543, 3))

        shift = 4 
        if self.augment is not None and random.random()>0.6 and len(xyz)>5:
            # TODO
            k0 = np.random.randint(0,len(xyz))
            k1 = np.random.randint(max(k0-shift,0), min(len(xyz), k0+shift))
            xyz = xyz - np.nan_to_num(xyz[k0:k0+1]) + np.nan_to_num(xyz[k1:k1+1])
        
        if self.augment is not None and random.random()>0.5:
            # randomly select another sample with the same label
            new_idx = random.choice(self.label_map[label])
            new_xyz = self.data[new_idx]['data'].reshape((-1, 543, 3))
            
            if random.random()>0.5:
                # mixup two samples with the same label
                l=min(len(xyz),len(new_xyz))
                xyz[:l,:,:] = (xyz[:l,:,:] + new_xyz[:l,:,:]) / 2
            elif random.random()>0.5:
                # random select another sample with the same label, shuffle the original coords with the delta of the two selected samples

                new_idx = random.choice(self.label_map[label])
                new_xyz2 = self.data[new_idx]['data'].reshape((-1, 543, 3))
                
                l=min(len(xyz),len(new_xyz),len(new_xyz2))
                xyz[:l,:,:] = xyz[:l,:,:] + new_xyz[:l,:,:] - new_xyz2[:l,:,:]
            else:
                # randomly replace the right hand / left hand / body part with the selected samples
                l=min(len(xyz),len(new_xyz))
                if random.random()>0.5:
                    xyz[:l,RHAND,:] = new_xyz[:l,RHAND,:]
                elif random.random()>0.5:            
                    xyz[:l,LHAND,:] = new_xyz[:l,LHAND,:]
                else:
                    xyz[:l,BODY,:] = new_xyz[:l,BODY,:]
            
            # randomly select a slice from the original sequence
            l = len(xyz)
            k1 = np.random.randint(0, 1+int(l*0.15))
            k2 = np.random.randint(0, 1+int(l*0.15))
            xyz = xyz[k1:len(xyz)-k2]

        elif self.augment is not None and random.random()>0.5:
            # randomly select another sample with the same label, use the start position of the original sample and the moving information of the selected sample to construct a new sample
            new_idx = random.choice(self.label_map[label])
            new_xyz = self.data[new_idx]['data'].reshape((-1, 543, 3))

            x0 = np.nan_to_num(xyz[:1,:,:])
            x_diff = new_xyz - np.nan_to_num(new_xyz[:1,:,:])

            xyz = x_diff + x0
            xyz[xyz==0] = np.nan

            l = len(xyz)
            k1 = np.random.randint(0, 1+int(l*0.15))
            k2 = np.random.randint(0, 1+int(l*0.15))
            xyz = xyz[k1:len(xyz)-k2]

        
        # only use the xy coords
        xyz = xyz[:,:,:2]
        
        if self.augment is not None and random.random()>0.8:
            # randomly resize the original sequence by interpolation
            l,dim,dim2 = xyz.shape
            b=range(l)
            f=interp1d(b,xyz,axis=0)
            step = np.random.uniform(low=0.5, high=2)
            new_b=list(np.arange(0,l-1,step))+[l-1]
            xyz = f(new_b)

        
        xyz_flat = xyz.reshape(-1,2)
        m = np.nanmean(xyz_flat,0).reshape(1,1,2)
    
        # apply coords normalization
        xyz = xyz - m #noramlisation to common maen
        xyz = xyz / np.nanstd(xyz_flat, 0).mean() 

        aug = 0
        if self.augment is not None:
            # applying data augmentation
            xyz = self.augment(xyz)
            aug = 1


        xyz = torch.from_numpy(xyz).float()
        xyz = pre_process(xyz,aug)[:self.maxlen]
        
        xyz[torch.isnan(xyz)] = 0

        # padding the sqeuence to a pre-defined max length
        data_pad = torch.zeros((self.maxlen, xyz.shape[1]), dtype=torch.float32)
        tot = xyz.shape[0]

        if tot <= self.maxlen:
            data_pad[:tot] = xyz
        else:
            data_pad[:] = xyz[:self.maxlen]

        if not self.training:
            # for validation
            return data_pad, label

        # if training, return a sample with two different augmentations
        if rec == False:
            data2 = self.__getitem__(idx, True)
            return data_pad, label, data2
        else:
            return data_pad


class ConcatDataset(Dataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


