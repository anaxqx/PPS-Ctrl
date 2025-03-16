import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.controlnet import ControlNetModel
from diffusers.utils import BaseOutput
from diffusers.models.unets.unet_2d_blocks import get_down_block


class PPSControlNetOutput(BaseOutput):
    """
    Modified output to include reconstructed conditioning.
    """
    down_block_res_samples: tuple
    mid_block_res_sample: torch.Tensor
    reconstructed_conditioning: torch.Tensor


class ControlNetConditioningDecoding(nn.Module):
    """
    Decodes the conditioning embeddings back to image-space conditions.
    """
    def __init__(self, conditioning_embedding_channels: int, conditioning_channels: int = 3, 
                 block_out_channels: tuple = (256, 96, 32, 16)):
        super().__init__()
        self.conv_in = nn.Conv2d(conditioning_embedding_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            self.blocks.append(nn.ConvTranspose2d(block_out_channels[i], block_out_channels[i + 1], kernel_size=4, stride=2, padding=1))
            self.blocks.append(nn.Conv2d(block_out_channels[i + 1], block_out_channels[i + 1], kernel_size=3, padding=1))

        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_channels, kernel_size=3, padding=1)

    def forward(self, embedding):
        embedding = F.silu(self.conv_in(embedding))
        for block in self.blocks:
            embedding = F.silu(block(embedding))
        return self.conv_out(embedding)


class PPSControlNet(ControlNetModel):
    """
    Modified ControlNetModel with additional decoding and custom behavior.
    """

    def __init__(self, *args, decoder: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = decoder  # Custom attribute

        if self.decoder:
            self.controlnet_cond_decoding = ControlNetConditioningDecoding(
                conditioning_embedding_channels=self.config.block_out_channels[0],
                block_out_channels=self.config.conditioning_embedding_out_channels[::-1],
                conditioning_channels=self.config.in_channels
            )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor, 
                controlnet_cond: torch.Tensor, conditioning_scale: float = 1.0, return_dict: bool = True, **kwargs):
        """
        Custom forward function with additional reconstruction handling.
        """

        # Run the original ControlNet forward pass
        controlnet_output = super().forward(sample, timestep, encoder_hidden_states, controlnet_cond, conditioning_scale, return_dict=False)

        # If decoder is enabled, reconstruct the conditioning image
        reconstructed_conditioning = None
        if self.decoder:
            reconstructed_conditioning = self.controlnet_cond_decoding(controlnet_cond)

        if not return_dict:
            return controlnet_output + (reconstructed_conditioning,)

        return PPSControlNetOutput(
            down_block_res_samples=controlnet_output[0],
            mid_block_res_sample=controlnet_output[1],
            reconstructed_conditioning=reconstructed_conditioning
        )

    @classmethod
    def from_pretrained_controlnet(cls, model_name: str, **kwargs):
        """
        Load from a pretrained ControlNet model and apply modifications.
        """
        base_model = ControlNetModel.from_pretrained(model_name, **kwargs)
        return cls(**base_model.config)