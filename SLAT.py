###############################################################################
#                                                                             #
# SLAT                                                                        #
# D. Schneider                                                                #
#                                                                             #
###############################################################################

###############################################################################
#                                                                             #
# Imports                                                                     #
#                                                                             #
###############################################################################
import lightning.pytorch as pl
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from huggingface_hub import PyTorchModelHubMixin


###############################################################################
#                                                                             #
# Attention Mask Submodule                                                    #
#                                                                             #
###############################################################################
class AttentionMask(torch.nn.Module):
    def __init__(self, timeWindowSize, features):
        super().__init__()
        globMask = torch.zeros((timeWindowSize, timeWindowSize))
        globMask[2:, 0] = globMask[2:, 0] + 1
        globMask[0, 2:] = globMask[0, 2:] + 1
        globMask[-1, 1] = 1
        globMask[1, -1] = 1
        band = torch.zeros((timeWindowSize, timeWindowSize))
        for i in range(band.shape[0]):
            for j in range(band.shape[1]):
                if i == j:
                    band[i, j] = 1
                if i > 0 and (i-1) == j:
                    band[i, j] = 1
                if j > 0 and (j-1) == i:
                    band[i, j] = 1
        sparse_att_mask_t = globMask + band
        self.register_buffer("sparse_att_mask_t", sparse_att_mask_t)

        globMask = torch.zeros((features, features))
        globMask[2:, 0] = globMask[2:, 0] + 1
        globMask[0, 2:] = globMask[0, 2:] + 1
        globMask[-1, 1] = 1
        globMask[1, -1] = 1
        band = torch.zeros((features, features))
        for i in range(band.shape[0]):
            for j in range(band.shape[1]):
                if i == j:
                    band[i, j] = 1
                if i > 0 and (i-1) == j:
                    band[i, j] = 1
                if j > 0 and (j-1) == i:
                    band[i, j] = 1
        sparse_att_mask_s = globMask + band
        self.register_buffer("sparse_att_mask_s", sparse_att_mask_s)

    def forward(self):
        return self.sparse_att_mask_t, self.sparse_att_mask_s


###############################################################################
#                                                                             #
# Positional Encoding Submodule                                               #
#                                                                             #
###############################################################################
class PositionalEncoding(torch.nn.Module):
    def __init__(self, length, depth):
        super().__init__()
        depth = depth/2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        pe = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)
        self.register_buffer('pe', torch.Tensor(pe))

    def forward(self, x):
        x = x + self.pe
        return x


###############################################################################
#                                                                             #
# Sensor Encoder Submodule                                                    #
#                                                                             #
###############################################################################
class SensorEncoder(torch.nn.Module):
    def __init__(self, dimv, dimatt, n_heads, drop):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.attn = torch.nn.MultiheadAttention(
            dimatt,
            n_heads,
            drop,
            batch_first=True
        )
        self.ln2 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.ffn1 = torch.nn.Linear(dimv, dimv)
        self.ffn2 = torch.nn.Linear(dimv, dimv)

    def forward(self, x, mask=None):
        a = self.ln1(x)
        a, _ = self.attn(a, a, a, attn_mask=mask)
        x = self.ln2(a + x)
        a = self.ffn2(torch.nn.ELU()(self.ffn1(x)))
        return x + a


###############################################################################
#                                                                             #
# Time Step Encoder Submodule                                                 #
#                                                                             #
###############################################################################
class TimeEncoder(torch.nn.Module):
    def __init__(self, dimv, dimatt, n_heads, drop):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.attn = torch.nn.MultiheadAttention(
            dimatt,
            n_heads,
            drop,
            batch_first=True
        )
        self.ln2 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.ffn1 = torch.nn.Linear(dimv, dimv)
        self.ffn2 = torch.nn.Linear(dimv, dimv)

    def forward(self, x, mask=None):
        a = self.ln1(x)
        a, _ = self.attn(a, a, a, attn_mask=mask)
        x = self.ln2(a + x)
        a = self.ffn2(torch.nn.ELU()(self.ffn1(x)))
        return x + a


###############################################################################
#                                                                             #
# Decoder Submodule                                                           #
#                                                                             #
###############################################################################
class Decoder(torch.nn.Module):
    def __init__(self, dimv, dimatt, n_heads, drop):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.attn1 = torch.nn.MultiheadAttention(
            dimatt,
            n_heads,
            drop,
            batch_first=True
        )
        self.ln2 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.attn2 = torch.nn.MultiheadAttention(
            dimatt,
            n_heads,
            drop,
            batch_first=True
        )
        self.ln3 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.ffn1 = torch.nn.Linear(dimv, dimv)
        self.ffn2 = torch.nn.Linear(dimv, dimv)

    def forward(self, x, enc):
        a = self.ln1(x)
        a, _ = self.attn1(a, a, a, key_padding_mask=None)
        x = self.ln2(a + x)
        a, _ = self.attn2(x, enc, enc, key_padding_mask=None)
        x = self.ln3(a + x)
        a = self.ffn2(torch.nn.ELU()(self.ffn1(x)))
        return x + a


###############################################################################
#                                                                             #
# SLAT Module                                                                 #
#                                                                             #
###############################################################################
class SLAT_LitModule(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.input_size = (42, 34)
        self.d_model = 64
        self.heads = 8
        self.nencoder = 4
        self.ndecoder = 2
        self.sparse_att_mask = torch.nn.ModuleList(
            [AttentionMask(self.input_size[0], self.input_size[1])]
        )
        self.dim_val = self.d_model
        self.dec_l = 4
        self.output_size = 1

        self.sensor_encoder = torch.nn.ModuleList(
            [SensorEncoder(self.dim_val, self.d_model, self.heads, 0)
             for _ in range(self.nencoder)]
        )

        self.time_encoder = torch.nn.ModuleList(
            [TimeEncoder(self.dim_val, self.d_model, self.heads, 0)
             for _ in range(self.nencoder)]
        )

        self.decoder = torch.nn.ModuleList(
            [Decoder(self.dim_val, self.d_model, self.heads, 0)
             for _ in range(self.ndecoder)]
        )

        self.pos_s = torch.nn.ModuleList(
            [PositionalEncoding(self.input_size[1], self.dim_val)]
        )

        self.pos_t = torch.nn.ModuleList(
            [PositionalEncoding(self.input_size[0], self.dim_val)]
        )

        self.encemb_s = torch.nn.Linear(self.input_size[0], self.dim_val)
        self.encemb_t = torch.nn.Linear(self.input_size[1], self.dim_val)
        self.encemb_d = torch.nn.Linear(self.input_size[1], self.dim_val)

        self.ln1 = torch.nn.LayerNorm(self.dim_val, eps=1e-5)
        self.out = torch.nn.Linear(self.dec_l*self.dim_val,
                                   self.output_size)

    def forward(self, x):
        x_time = x
        x_sensor = torch.transpose(x, dim0=1, dim1=2)

        samt, sams = self.sparse_att_mask[0]()

        te = self.time_encoder[0](self.pos_t[0](self.encemb_t(x_time)),
                                  samt)
        se = self.sensor_encoder[0](self.pos_s[0](self.encemb_s(x_sensor)),
                                    sams)

        for t_enc in self.time_encoder[1:]:
            te = t_enc(te, samt)

        for s_enc in self.sensor_encoder[1:]:
            se = s_enc(se, sams)

        p = torch.cat((se, te), dim=1)
        p = self.ln1(p)

        d = self.decoder[0](self.encemb_d(x_time[:, -self.dec_l:, :]), p)
        for d_enc in self.decoder[1:]:
            d = d_enc(d, p)

        x = self.out(torch.nn.ReLU()(d.flatten(start_dim=1)))

        return x

    def training_step(self, batch, batch_idx):
        _, loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train_RMSE', loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_RMSE', loss, sync_dist=True)

        # Let's return preds to use it in a custom callback
        return preds

    def test_step(self, batch, batch_idx):
        _, loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('test_RMSE', loss, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5)
        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": 'val_RMSE'}

    def _get_preds_loss_accuracy(self, batch):
        x, y = batch
        preds = self(x)
        loss = torch.sqrt(torch.nn.MSELoss()(preds, y))
        return preds, loss
