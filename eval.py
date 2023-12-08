import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Params to set for eval:

    # algorithm should be the directory name in the data_root and maps_root folder. Same as "Tag" from test.py
algorithm = 'GCPANet30'
datasets = ['SOD', 'DUTS', 'ECSSD', 'HKU-IS', 'PASCAL-S', 'DUT-OMRON']
data_root = '/media/nvme2/expansion/leeor/ece-project/data/'
maps_root = '/media/nvme2/expansion/leeor/ece-project/pred_maps/'

def Fmeasure(sMap, gtMap, gtsize):
    sumLabel = 2 * np.mean(sMap)
    sumLabel = min(sumLabel, 1)

    Label3 = np.zeros(gtsize, dtype=bool)  # Ensure Label3 is a boolean array
    Label3[sMap >= sumLabel] = True

    # If gtMap is not already boolean, convert it to boolean
    if not issubclass(gtMap.dtype.type, np.bool_):
        gtMap = gtMap > 0.5

    LabelAnd = Label3 & gtMap

    NumRec = np.sum(Label3)
    NumAnd = np.sum(LabelAnd)
    num_obj = np.sum(gtMap)

    if NumAnd == 0:
        PreFtem = RecallFtem = FmeasureF = 0
    else:
        PreFtem = NumAnd / NumRec
        RecallFtem = NumAnd / num_obj
        FmeasureF = (1.3 * PreFtem * RecallFtem) / (0.3 * PreFtem + RecallFtem)

    return PreFtem, RecallFtem, FmeasureF


def CalPR(smapImg, gtImg):
    # Check if ground truth is logical (binary), if not, convert
    if not issubclass(gtImg.dtype.type, np.bool_):
        gtImg = gtImg[:, :, 0] > 128

    # Check if saliency map and ground truth have the same size
    if smapImg.shape != gtImg.shape:
        raise ValueError('Saliency map and ground truth mask have different sizes')

    # Check if ground truth has any foreground pixel
    gtPxlNum = np.sum(gtImg)
    if gtPxlNum == 0:
        raise ValueError('No foreground region is labeled')

    # Calculate histograms
    targetHist, _ = np.histogram(smapImg[gtImg], bins=256, range=(0, 255))
    nontargetHist, _ = np.histogram(smapImg[~gtImg], bins=256, range=(0, 255))

    # Flip and cumulate histograms
    targetHist = np.flipud(targetHist).cumsum()
    nontargetHist = np.flipud(nontargetHist).cumsum()

    # Calculate precision and recall
    precision = targetHist / (targetHist + nontargetHist + np.finfo(float).eps)
    recall = targetHist / gtPxlNum

    # Handle potential NaNs in precision
    if np.any(np.isnan(precision)):
        print('Warning: There exists NAN in precision, this is because your saliency map does not range from 0 to 255')

    return precision, recall


# Initialize a dictionary to store precision and recall for each dataset
dataset_pr_values = {dataset: {'precisions': [], 'recalls': []} for dataset in datasets}

def run_eval(savefigs = ['DUTS', 'ECSSD', 'DUT-OMRON']):
    for alg in [algorithm]:
        print(f'Model: {alg}')
        for dataset in datasets:
            print(f'\tDataset: {dataset}')
            predpath = os.path.join(maps_root, alg, dataset)
            maskpath = os.path.join(data_root, dataset, 'mask')
            if not os.path.exists(predpath):
                continue

            test_file = 'test_poolnet.txt' if alg == 'PoolNet' and dataset == 'HKU-IS' else 'test.txt'
            with open(os.path.join(data_root, dataset, test_file), 'r') as file:
                names = file.read().splitlines()

            # Initialize metrics
            mae = fm = prec = rec = wfm = sm = em = 0
            results = []
            ALLPRECISION = []
            ALLRECALL = []
            file_num = []

            for name in names:
                fgpath = os.path.join(predpath, f'{name}_sal_fuse.png') if alg == 'PoolNet' else os.path.join(predpath, f'{name}.png')
                if not os.path.exists(fgpath):
                    fgpath = os.path.join(predpath, f'{name}.jpg')
                    if not os.path.exists(fgpath):
                        continue

                fg = Image.open(fgpath).convert('L')
                gtpath = os.path.join(maskpath, f'{name}.png')
                if not os.path.exists(gtpath):
                    gtpath = os.path.join(maskpath, f'{name}.jpg')
                    if not os.path.exists(gtpath):
                        continue

                gt = Image.open(gtpath).convert('L')
                fg = fg.resize(gt.size)
                fg = np.array(fg) / 255.0
                gt = np.array(gt) / 255.0

                if np.max(fg) == 0 or np.max(gt) == 0:
                    continue

                gt = np.where(gt >= 0.5, 1, 0).astype(bool)

                # Replace these with actual function calls
                # score1 = MAE(fg, gt)
                score2, score3, score4 = Fmeasure(fg, gt, gt.shape)
                # score5 = wFmeasure(fg, gt)
                # score6 = Smeasure(fg, gt)
                # score7 = Emeasure(fg, gt)

                # Update metrics
                # mae += score1
                fm += score4
                # wfm += score5
                # sm += score6
                # em += score7
                # results.append([name, score1, score4, score5, score6, score7])
                results.append([name,score4])
                precision, recall = CalPR(fg * 255, gt)
                ALLPRECISION.append(precision)
                ALLRECALL.append(recall)
                file_num.append(True)

            # Calculate mean values
            prec = np.mean(ALLPRECISION, axis=0)
            rec = np.mean(ALLRECALL, axis=0)
            maxF = np.max(1.3 * prec * rec / (0.3 * prec + rec + np.finfo(float).eps))

            dataset_pr_values[dataset]['precisions'].extend(np.mean(ALLPRECISION, axis=0))
            dataset_pr_values[dataset]['recalls'].extend(np.mean(ALLRECALL, axis=0))


    for dataset in datasets:
        precisions = np.array(dataset_pr_values[dataset]['precisions'])
        recalls = np.array(dataset_pr_values[dataset]['recalls'])

        # Sorting the values for plotting
        sort_order = recalls.argsort()
        recalls_sorted = recalls[sort_order]
        precisions_sorted = precisions[sort_order]
        from scipy import interpolate

        # Assuming recalls_sorted and precisions_sorted are sorted recall and precision values
        recall_interp = np.linspace(1e-3, 1, 100)  # 100 points for interpolation
        precision_interp = interpolate.interp1d(recalls_sorted, precisions_sorted, kind='next')(recall_interp)

        # Plotting the PR curve for each dataset
        plt.figure()
        plt.step(recall_interp, precision_interp, where='post')
        plt.ylim((0,1))
        plt.xlim((0,1))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {dataset}')
        if dataset in savefigs:
            plt.savefig(f'{dataset}_PR.pdf')
        plt.show()

if __name__ == '__main__': 
    run_eval()
