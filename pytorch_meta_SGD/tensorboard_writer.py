from tensorboardX import SummaryWriter

class TensorBoardWriter(object):

    def __init__(self, folder):
        self.writer = SummaryWriter(folder)
        self.export_json_path = folder + "/all_scalars.json"

    def record_support_loss(self, tensor, iter:int):
        self.writer.add_scalar("data/train_support_loss", tensor, iter)

    def recorad_query_loss(self, tensor, iter:int):
        self.writer.add_scalar("data/train_query_loss", tensor,  iter)

    def record_tot_loss(self, tensor, iter:int):
        self.writer.add_scalar("data/total_loss", tensor, iter)

    def record_val_accuracy(self, tensor, iter:int):
        self.writer.add_scalar("data/val_accuracy", tensor, iter)

    def record_accuracy_support(self, tensor, iter:int):
        self.writer.add_scalar("data/train_accuracy_support", tensor, iter)

    def record_accuracy_query(self, tensor, iter: int):
        self.writer.add_scalar("data/train_accuracy_query", tensor, iter)

    def record_accuracy_two_way(self, tensor, iter: int):
        self.writer.add_scalar("data/train_accuracy_two_way", tensor, iter)


    def record_val_two_accuracy(self, tensor, iter:int):
        self.writer.add_scalar("data/val_two_accuracy", tensor, iter)

    def record_val(self, val_name, tensor, iter:int):
        self.writer.add_scalar(val_name, tensor, iter)

    def record_train_accuracy(self, tensor, iter:int):
        self.writer.add_scalar("data/train_accuracy", tensor, iter)

    def export_json(self):
        self.writer.export_scalars_to_json(self.export_json_path)

    def close(self):
        self.writer.close()