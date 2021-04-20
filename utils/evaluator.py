import torch


class Evaluation:
    def __init__(self, predictions, labels):
        self.predictions = predictions
        self.labels = labels
        self.tp = (predictions == 1) * (labels == 1)

        self.tp_num = len(torch.nonzero(self.tp))
        self.tn = (predictions != 1) * (labels != 1)
        self.tn_num = len(torch.nonzero(self.tn))
        self.fp = (predictions == 1) * (labels != 1)
        self.fp_num = len(torch.nonzero(self.fp))
        self.fn = (predictions != 1) * (labels == 1)
        self.fn_num = len(torch.nonzero(self.fn)) #torch.nonzero(self.fn).shape.item()
        self.total = len(labels)



        self.precision = self.tp_num / (self.tp_num + self.fp_num) if self.tp_num + self.fp_num != 0 else 0
        self.recall = self.tp_num / (self.tp_num + self.fn_num) if self.tp_num + self.fn_num != 0 else 0


    def get_fp(self):
        return torch.nonzero(self.fp).squeeze()

    def get_tp(self):
        return torch.nonzero(self.tp).squeeze()

    def get_tn(self):
        return torch.nonzero(self.tn).squeeze()

    def get_fn(self):
        return torch.nonzero(self.fn).squeeze()

    def get_accuracy(self):
        return (self.tp_num + self.tn_num) / self.total

    def get_precision(self):
        return self.precision

    def get_recall(self):
        return self.recall

    def get_f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0


