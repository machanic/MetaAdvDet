from collections import defaultdict
import os
import pickle

class Fingerprints:
    def __init__(self):
        self.dx = None
        self.y = None
        self.dy = None

class Example:
    def __init__(self, x, yhat, y_class):
        self.id = None

        self.x = x
        self.yhat = yhat
        self.y_class = y_class
        self.is_legal = None
        self.dxs = None
        self.yhat_p = None
        self.diff = None
        self.diff_norm = None
        self.y_class_with_fp = None

class Stats:
    def __init__(self, tau=0, name="", ds_name=""):
        self.name = name
        self.ds_name = ds_name
        self.tau = tau

        self.ids = set()
        self.ids_correct = set()
        self.ids_correct_fp = set()
        self.ids_agree = set()

        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0

        # Legal = there is a fingerprint match below threshold tau
        self.ids_legal = set()

        self.counts = defaultdict(lambda: 0)
        self.counts_legal = defaultdict(lambda: 0)
        self.counts_correct = defaultdict(lambda: 0)



        # Total number of examples
        self.i = 0

    def compute_counts(self, ids_correct=None):
        i = self.ids
        c = self.ids_correct
        c_fp = self.ids_correct_fp
        l = self.ids_legal
        a = self.ids_agree
        aa = set.intersection(
            self.ids_correct, self.ids_correct_fp)  # ground truth == predict label == px predict label

        if ids_correct: # use to only look at examples in ids_correct, i.e. the ones where f(x) was correct for test x.
            i = set.intersection(i, ids_correct)
            c = set.intersection(c, ids_correct)
            c_fp = set.intersection(c_fp, ids_correct)
            l = set.intersection(l, ids_correct)
            a = set.intersection(a, ids_correct)
            aa = set.intersection(aa, ids_correct)

        # Reject if not legal: model output does not match any fingerprint at threshold tau.
        # when does argmax f(x) == argmax f(x+dx)
        # when does y* == argmax f(x) == argmax f(x+dx)
        return {"num" : len(i),
                "num_correct" : len(c),
                "num_correct_fp" : len(c_fp),
                "num_legal" : len(l),
                "num_reject" : len(i) - len(l),
                "num_agree" : len(a),
                "num_all_agree" : len(aa)}

    def show(self, d):
        n = d["num"]
        if n <= 0:
            print("Empty set!")
            return
        for k,v in d.items():
            print("{:<20}: {:3.2f}% ({} / {})".format(k, v / n * 100, v, n))

    def write_dict(self, d, fn, log_dir, name):
        path = os.path.join(log_dir, "{}-{}-{}-tau_{:.4f}.pkl".format(name, self.ds_name, fn, self.tau))
        print("Saving stats in", path)
        pickle.dump(d, open(path, "wb"))

    def dump(self, log_dir, name):

        dicts = [self.counts, self.counts_legal, self.counts_correct]
        fns = ["args", "counts", "counts_legal", "counts_correct"]

        for result, fn in zip(dicts, fns):
            self.write_dict(result, fn, log_dir, name)

