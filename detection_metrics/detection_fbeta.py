import numpy as np
from detection_metrics.utils.bbox import jaccard


class Fbeta:
    def __init__(self, n_class, beta=1, overlap_threshold=0.5):
        self.n_class = n_class
        self.beta = beta
        self.iou_thresh = overlap_threshold

    def f_beta(self, tp, fp, fn):
        if tp == 0 and fp == 0 and fn == 0:
            return -1
        return (1 + self.beta ** 2) * tp / ((1 + self.beta ** 2) * tp + (self.beta ** 2) * fn + fp)

    def evaluate_dataset(self, pred_bb_list, gt_bb_list, pred_classes_list=None, gt_classes_list=None):

        if gt_classes_list is None or pred_classes_list is None:
            assert self.n_class == 1, f"Classes are not provided though n_classes = {self.n_class}"

        assert (len(pred_bb_list) == len(gt_bb_list))
        if pred_classes_list is not None:
            assert (len(pred_bb_list) == len(pred_classes_list))
        if gt_classes_list is not None:
            assert (len(gt_bb_list) == len(gt_classes_list))
        print("If -1 returned than tp, fp, fn = 0! So dont include thouse objects into metric calculations!")

        out = []
        for i in range(len(pred_bb_list)):
            pred = np.array(pred_bb_list[i])
            gt = np.array(gt_bb_list[i])

            pred_cls = np.zeros((len(pred)))
            if pred_classes_list is not None:
                pred_cls = np.array(pred_classes_list[i])

            gt_cls = np.zeros((len(gt)))
            if gt_classes_list is not None:
                gt_cls = np.array(gt_classes_list[i])

            k = self.evaluate(pred, pred_cls, gt, gt_cls)
            out.append(k)
        out = np.array(out)

        mean_f_betas_per_class = []
        for i in range(self.n_class):
            x = out[:, i]
            x = x[x != -1]
            mean_f_betas_per_class.append(x.mean())
        return out, mean_f_betas_per_class

    def evaluate(self, pred_bb, pred_classes, gt_bb, gt_classes):
        f_beta_per_class = []
        if len(pred_bb) == 0 and len(gt_bb) == 0:
            return [-1] * 3
        elif len(pred_bb) == 0:
            for i in range(self.n_class):
                gt_number = np.sum(gt_classes == i)
                fn = gt_number
                tp = 0
                fp = 0
                f_beta = self.f_beta(tp, fp, fn)
                f_beta_per_class.append(f_beta)
            return f_beta_per_class

        elif len(gt_bb) == 0:
            for i in range(self.n_class):
                pred_number = np.sum(pred_classes == i)
                fp = pred_number
                tp = 0
                fn = 0
                f_beta = self.f_beta(tp, fp, fn)
                f_beta_per_class.append(f_beta)
            return f_beta_per_class

        IoUmask = self.compute_iou_mask(pred_bb, gt_bb, self.iou_thresh)

        for i in range(self.n_class):
            gt_number = np.sum(gt_classes == i)
            pred_mask = (pred_classes == i)
            pred_number = np.sum(pred_mask)
            if pred_number == 0:
                fp = gt_number
                tp = 0
                fn = 0
            else:
                IoU1 = IoUmask[pred_mask, :]
                mask = IoU1[:, gt_classes == i]

                tp = Fbeta.compute_true_positive(mask)
                fp = pred_number - tp
                fn = gt_number - tp

            f_beta = self.f_beta(tp, fp, fn)
            f_beta_per_class.append(f_beta)
        return f_beta_per_class

    @staticmethod
    def compute_iou_mask(prediction, gt, overlap_threshold):
        IoU = jaccard(prediction, gt)
        # for each prediction select gt with the largest IoU and ignore the others
        for i in range(len(prediction)):
            maxj = IoU[i, :].argmax()
            IoU[i, :maxj] = 0
            IoU[i, (maxj + 1):] = 0
        # make a mask of all "matched" predictions vs gt
        return IoU >= overlap_threshold

    @staticmethod
    def compute_true_positive(mask):
        # sum all gt with prediction of its class
        return np.sum(mask.any(axis=0))
