from tensorboardX import SummaryWriter

class TensorBoardWriter(object):

    def __init__(self, folder):
        self.writer = SummaryWriter(folder)
        self.export_json_path = folder + "/all_scalars.json"

    def record_trn_support_loss(self, tensor, iter:int):
        self.writer.add_scalar("data/trn_support_loss", tensor, iter)

    def record_trn_support_acc(self, tensor, iter:int):
        self.writer.add_scalar("data/trn_support_acc", tensor, iter)
    def record_trn_query_acc(self, tensor, iter:int):
        self.writer.add_scalar("data/trn_query_acc", tensor, iter)

    def record_trn_query_loss(self, tensor, iter:int):
        self.writer.add_scalar("data/trn_query_loss", tensor, iter)

    def record_val_support_loss(self, tensor, iter:int):
        self.writer.add_scalar("data/val_support_loss", tensor, iter)

    def record_val_query_loss(self, tensor, iter:int):
        self.writer.add_scalar("data/val_query_loss", tensor, iter)

    def record_val_support_acc(self, tensor, iter:int):
        self.writer.add_scalar("data/val_support_acc", tensor, iter)

    def record_val_query_acc(self, tensor, iter:int):
        self.writer.add_scalar("data/val_query_acc", tensor, iter)

    def export_json(self):
        self.writer.export_scalars_to_json(self.export_json_path)

    def close(self):
        self.writer.close()