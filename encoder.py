import torch
import torch.nn as nn

import resnet


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# OUT_DIM = {2: 39, 4: 35, 6: 31}
OUT_DIM = {2: 39, 4: 57, 6: 31}


class BasePixelEncoder(nn.Module):
    """Base Convolutional encoder of pixels observations."""

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram("train_encoder/%s_hist" % k, v, step)
            if len(v.shape) > 2:
                L.log_image("train_encoder/%s_img" % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param("train_encoder/conv%s" % (i + 1), self.convs[i], step)
        L.log_param("train_encoder/fc", self.fc, step)
        L.log_param("train_encoder/ln", self.ln, step)

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""

        for (
            (src_module_name, src_module_param),
            (trg_module_name, trg_module_param),
        ) in zip(source.named_parameters(), self.named_parameters()):
            # Note that we tie the fc layer of the encoder as well but that layer isnt used.
            if "encoder" in src_module_name:
                if "weight" in src_module_name or "bias" in src_module_name:
                    trg_module_param = src_module_param


class PixelEncoder(BasePixelEncoder):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        assert len(obs_shape) in [3, 4]

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def _encode_using_convs(self, obs):
        conv = torch.relu(self.convs[0](obs))
        self.outputs["conv1"] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs["conv%s" % (i + 1)] = conv

        h = conv.reshape(conv.size(0), -1)
        return h

    def forward_conv(self, obs):
        obs = obs / 255.0
        obs_shape = obs.shape
        batched_obs = obs.reshape(obs_shape[0] * obs.shape[1], *obs.shape[2:])
        output = self._encode_using_convs(batched_obs)
        return output.reshape(obs_shape[0], obs_shape[1], -1).mean(dim=1)

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs["fc"] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs["ln"] = h_norm

        out = torch.tanh(h_norm)
        self.outputs["tanh"] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])


class ResNetEncoder(BasePixelEncoder):
    """Convolutional encoder of pixels observations."""

    def __init__(
        self,
        obs_shape,
        feature_dim,
        num_layers=2,
        num_filters=32,
        num_stacked_frames: int = 3,
    ):
        super().__init__()

        self.encoder = resnet._resnet_encoder(
            block=resnet.Bottleneck, layers=[2, 2, 2, 2], weights=None, progress=False
        )
        self.num_stacked_frames = num_stacked_frames
        assert len(obs_shape) == 4

        self.encoder_output_dim = 2048
        self.feature_dim = feature_dim
        self.fc = nn.Linear(self.encoder_output_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward_conv(self, obs):
        obs = obs / 255.0
        obs_shape = obs.shape
        batched_obs = obs.reshape(obs_shape[0] * obs.shape[1], *obs.shape[2:])
        output = self.encoder(batched_obs)
        return output.reshape(obs_shape[0], obs_shape[1], self.encoder_output_dim).mean(
            dim=1
        )

    def forward(self, obs, detach=False):

        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)

        h_norm = self.ln(h_fc)

        out = torch.tanh(h_norm)

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""

        for (
            (src_module_name, src_module_param),
            (trg_module_name, trg_module_param),
        ) in zip(source.named_parameters(), self.named_parameters()):
            # Note that we tie the fc layer of the encoder as well but that layer isnt used.
            if "encoder" in src_module_name:
                if "weight" in src_module_name or "bias" in src_module_name:
                    trg_module_param = src_module_param


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {
    "pixel": PixelEncoder,
    "identity": IdentityEncoder,
    "resnet": ResNetEncoder,
}


def make_encoder(encoder_type, obs_shape, feature_dim, num_layers, num_filters):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )
