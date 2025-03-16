import torch
from diffusers.pipelines.controlnet import StableDiffusionControlNetPipeline
from diffusers.utils import logging
from pps_ctrl import PPSControlNet

logger = logging.get_logger(__name__)  # Logging setup

class PPSStableDiffusionControlNetPipeline(StableDiffusionControlNetPipeline):
    """
    A modified version of the Stable Diffusion ControlNet pipeline that supports `PPSControlNet`,
    which returns an additional `reconstructed_conditioning` output.
    """

    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, controlnet):
        """
        Initializes the modified PPS ControlNet pipeline.
        """
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, controlnet)
        
        if not isinstance(controlnet, PPSControlNet):
            raise ValueError("This pipeline requires a `PPSControlNet` model.")

    def __call__(self, prompt, image, num_inference_steps=50, guidance_scale=7.5, conditioning_scale=1.0, **kwargs):
        """
        Runs the modified pipeline with support for reconstructed conditioning.
        """
        # Run the original ControlNet processing
        controlnet_output = self.controlnet(
            sample=image,
            timestep=torch.tensor([num_inference_steps]),
            encoder_hidden_states=self.encode_prompt(prompt),
            controlnet_cond=image,
            conditioning_scale=conditioning_scale,
            return_dict=True
        )

        # Extract the extra output from PPSControlNet
        reconstructed_conditioning = controlnet_output.reconstructed_conditioning

        logger.info(f"Reconstructed conditioning shape: {reconstructed_conditioning.shape if reconstructed_conditioning is not None else 'None'}")

        # Proceed with the original pipeline flow
        output = super().__call__(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )

        return output, reconstructed_conditioning  # Return additional output