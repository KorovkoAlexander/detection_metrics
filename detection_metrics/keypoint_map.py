import numpy as np
from detection_metrics.ap_accumulator import APAccumulator


class MAP:
    def __init__(self, n_class, pr_samples=11, overlap_threshold=0.5):
        """
        Running computation of average precision of n_class in a bounding box + classification task
        :param n_class:             quantity of class
        :param pr_samples:          quantification of threshold for pr curve
        :param overlap_threshold:   minimum overlap threshold
        """
        self.n_class = n_class
        self.overlap_threshold = overlap_threshold
        self.pr_scale = np.linspace(0, 1, pr_samples)
        self.total_accumulators = []
        self.reset_accumulators()

    def reset_accumulators(self):
        """
        Reset the accumulators state
        TODO this is hard to follow... should use a better data structure
        total_accumulators : list of list of accumulators at each pr_scale for each class
        :return:
        """
        self.total_accumulators = []
        for i in range(len(self.pr_scale)):
            class_accumulators = []
            for j in range(self.n_class):
                class_accumulators.append(APAccumulator())
            self.total_accumulators.append(class_accumulators)

    def evaluate_dataset(self,
                         pred_keypoints_list,
                         gt_keypoints_list,
                         pred_conf_list,
                         img_shapes,
                         pred_classes_list=None,
                         gt_classes_list=None):
        if gt_classes_list is None or pred_classes_list is None:
            assert self.n_class == 1, f"Classes are not provided though n_classes = {self.n_class}"

        assert (len(pred_keypoints_list) == len(gt_keypoints_list))
        assert (len(pred_keypoints_list) == len(pred_conf_list))
        assert (len(pred_keypoints_list) == len(img_shapes))
        if pred_classes_list is not None:
            assert (len(pred_keypoints_list) == len(pred_classes_list))
        if gt_classes_list is not None:
            assert (len(gt_keypoints_list) == len(gt_classes_list))

        for i in range(len(pred_keypoints_list)):
            pred = np.array(pred_keypoints_list[i])
            gt = np.array(gt_keypoints_list[i])
            pred_sc = np.array(pred_conf_list[i])
            img_shape = img_shapes[i]

            pred_cls = np.zeros((len(pred)))
            if pred_classes_list is not None:
                pred_cls = np.array(pred_classes_list[i])

            gt_cls = np.zeros((len(gt)))
            if gt_classes_list is not None:
                gt_cls = np.array(gt_classes_list[i])

            if len(gt) == 0:
                continue

            # print(pred.shape, pred_cls.shape, pred_sc.shape, gt.shape, gt_cls.shape)
            self.evaluate(img_shape, pred, gt, pred_sc, pred_cls, gt_cls)

    def evaluate(self, img_shape, pred_keypoints, gt_keypoints, pred_conf, pred_classes=None, gt_classes=None):
        """
        Update the accumulator for the running mAP evaluation.
        For exemple, this can be called for each images
        :param pred_keypoints: (np.array)Predicted Bounding Boxes [x1, y1, x2, y2] :     Shape [n_pred, 2]
        :param img_shape: (np.array) Image shape [w, h] :     Shape [2]
        :param pred_classes: (np.array)  Predicted Classes :                             Shape [n_pred]
        :param pred_conf: (np.array)     Predicted Confidences [0.-1.] :                 Shape [n_pred]
        :param gt_keypoints: (np.array)  Ground Truth Bounding Boxes [x1, y1, x2, y2] :  Shape [n_gt, 2]
        :param gt_classes: (np.array)    Ground Truth Classes :                          Shape [n_gt]
        :return:
        """

        if pred_classes is None:
            pred_classes = np.zeros((len(pred_keypoints)))
        if gt_classes is None:
            gt_classes = np.zeros((len(gt_keypoints)))

        if pred_keypoints.ndim == 1:
            pred_keypoints = np.repeat(pred_keypoints[:, np.newaxis], 2, axis=1)

        img_diag = np.sqrt(np.power(img_shape, 2).sum())
        match_thresh = img_diag * 0.05

        matchings = None
        if len(pred_keypoints) > 0:
            matchings = self.compute_matchings(pred_keypoints, gt_keypoints, match_thresh)
        for accumulators, r in zip(self.total_accumulators, self.pr_scale):
            self.evaluate_(matchings, accumulators, pred_classes, pred_conf, gt_classes, r)

    @staticmethod
    def evaluate_(matchings, accumulators, pred_classes, pred_conf, gt_classes, confidence_threshold):
        pred_classes = pred_classes.astype(np.int)
        gt_classes = gt_classes.astype(np.int)

        for i, acc in enumerate(accumulators):
            gt_number = np.sum(gt_classes == i)
            pred_mask = np.logical_and(pred_classes == i, pred_conf >= confidence_threshold)
            pred_number = np.sum(pred_mask)
            if pred_number == 0:
                acc.inc_not_predicted(gt_number)
                continue

            matchings1 = matchings[pred_mask, :]
            mask = matchings1[:, gt_classes == i]

            tp = MAP.compute_true_positive(mask)
            fp = pred_number - tp
            fn = gt_number - tp
            acc.inc_good_prediction(tp)
            acc.inc_not_predicted(fn)
            acc.inc_bad_prediction(fp)

    @staticmethod
    def compute_matchings(prediction, gt, thresh):
        costs = np.zeros((len(prediction), len(gt)))
        for i, p in enumerate(prediction):
            for j, g in enumerate(gt):
                costs[i, j] = np.linalg.norm(p-g)

        for i in range(len(prediction)):
            maxj = costs[i, :].argmin()
            costs[i, :maxj] = float('inf')
            costs[i, (maxj + 1):] = float('inf')
        # make a mask of all "matched" predictions vs gt
        return costs <= thresh

    @staticmethod
    def compute_true_positive(mask):
        # sum all gt with prediction of its class
        return np.sum(mask.any(axis=0))

    def compute_ap(self, precisions, recalls):
        """
        Compute average precision of a particular classes (cls_idx)
        :param cls:
        :return:
        """
        previous_recall = 0
        average_precision = 0
        for precision, recall in zip(precisions[::-1], recalls[::-1]):
            average_precision += precision * (recall - previous_recall)
            previous_recall = recall
        return average_precision

    def compute_precision_recall_(self, class_index, interpolated=True):
        precisions = []
        recalls = []
        for acc in self.total_accumulators:
            precisions.append(acc[class_index].precision)
            recalls.append(acc[class_index].recall)

        if interpolated:
            interpolated_precision = []
            for precision in precisions:
                last_max = 0
                if interpolated_precision:
                    last_max = max(interpolated_precision)
                interpolated_precision.append(max(precision, last_max))
            precisions = interpolated_precision
        return precisions, recalls

    def mAP(self, interpolated=True):
        mean_average_precision = []
        for i in range(self.n_class):
            precisions, recalls = self.compute_precision_recall_(i, interpolated)
            average_precision = self.compute_ap(precisions, recalls)
            mean_average_precision.append(average_precision)

        return mean_average_precision
