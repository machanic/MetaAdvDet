from tensorboardX import SummaryWriter

class TensorBoardWriter(object):

    def __init__(self, folder, data_prefix):
        self.writer = SummaryWriter(folder)
        self.export_json_path = folder + "/all_scalars.json"
        self.data_prefix = data_prefix

    def record_trn_support_loss(self, tensor, iter:int):
        self.writer.add_scalar("{}/trn_support_loss".format(self.data_prefix), tensor, iter)

    def record_trn_support_acc(self, tensor, iter:int):
        self.writer.add_scalar("{}/trn_support_acc".format(self.data_prefix), tensor, iter)

    def record_trn_query_acc(self, tensor, iter:int):
        self.writer.add_scalar("{}/trn_query_acc".format(self.data_prefix), tensor, iter)

    def record_trn_query_loss(self, tensor, iter:int):
        self.writer.add_scalar("{}/trn_query_loss".format(self.data_prefix), tensor, iter)

    def record_val_support_loss(self, tensor, iter:int):
        self.writer.add_scalar("{}/val_support_loss".format(self.data_prefix), tensor, iter)

    def record_val_query_loss(self, tensor, iter:int):
        self.writer.add_scalar("{}/val_query_loss".format(self.data_prefix), tensor, iter)

    def record_val_support_acc(self, tensor, iter:int):
        self.writer.add_scalar("{}/val_support_acc".format(self.data_prefix), tensor, iter)

    def record_val_query_acc(self, tensor, iter:int):
        self.writer.add_scalar("{}/val_query_acc".format(self.data_prefix), tensor, iter)

    def record_val_support_twoway_acc(self, tensor, iter:int):
        self.writer.add_scalar("{}/val_support_2way_acc".format(self.data_prefix), tensor, iter)

    def record_val_query_twoway_acc(self, tensor, iter:int):
        self.writer.add_scalar("{}/val_query_2way_acc".format(self.data_prefix), tensor, iter)

    def record_trn_support_twoway_acc(self, tensor, iter:int):
        self.writer.add_scalar("{}/trn_support_2way_acc".format(self.data_prefix), tensor, iter)

    def record_trn_query_twoway_acc(self, tensor, iter:int):
        self.writer.add_scalar("{}/trn_query_2way_acc".format(self.data_prefix), tensor, iter)

    def export_json(self):
        self.writer.export_scalars_to_json(self.export_json_path)

    def close(self):
        self.writer.close()