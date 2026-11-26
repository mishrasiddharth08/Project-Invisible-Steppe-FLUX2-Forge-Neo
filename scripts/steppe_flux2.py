import sys
from pathlib import Path

import gradio as gr
import torch

from backend import memory_management as memory
from backend.nn.vae import AutoencoderKLFlux2
from backend.patcher.base import ModelPatcher
from modules import paths, script_callbacks, scripts, shared
from pi_steppe import UpscaleSettings, load_upscaler_file


MODELS = {
    "Standard (1 step)": "steppe_flux_2_200M_1step.safetensors",
    "Lite (1 step)": "steppe_flux_2_20M_1step.safetensors",
    "Base (30 steps)": "steppe_flux_2_200M_base.safetensors",
}
REVISION = "ed6867c62927f79fc6d1d5b3ab0aad33d1e2aa71"


def model_path(choice):
    filename = MODELS[choice]
    directory = Path(paths.models_path) / "Steppe"
    path = directory / filename
    if path.is_file():
        return str(path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download("TensorForger/Steppe-Flux.2", filename,
                           revision=REVISION, local_dir=str(directory))


def incompatibility(p, samples):
    if getattr(p, "image_mask", None) is not None or getattr(p, "overlay_images", None):
        return "masked img2img uses the original image dimensions"
    if getattr(samples, "already_decoded", False):
        return "this generation path already decoded the image"
    if not isinstance(samples, torch.Tensor) or samples.ndim != 4 or samples.shape[1] != 128:
        return "expected a packed 128-channel FLUX.2 image latent"
    vae = p.sd_model.forge_objects.vae
    stage = vae.first_stage_model
    fmt = getattr(stage, "latent_format", None)
    if (not isinstance(stage, AutoencoderKLFlux2) or stage.mugen
            or getattr(vae, "downscale_ratio", None) != 16
            or getattr(fmt, "taesd_decoder_name", None) != "taef2_decoder"
            or getattr(fmt, "latent_channels", None) != 128):
        return "the active VAE does not identify a packed FLUX.2 latent format"
    return None


class SteppeFlux2(scripts.Script):
    def title(self):
        return "Steppe FLUX.2"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Steppe FLUX.2 latent upscale", open=False):
            enabled = gr.Checkbox(label="Enable Steppe", value=False)
            scale = gr.Radio(["2x", "4x"], value="2x", label="Upscale")
            model = gr.Dropdown(list(MODELS), value=next(iter(MODELS)), label="Steppe model")
            sharpness = gr.Slider(0, 0.95, value=0.9, step=0.01, label="Fine detail")
        return [enabled, scale, model, sharpness]

    @torch.inference_mode()
    def post_sample(self, p, ps, enabled=False, scale="2x", model="Standard (1 step)", sharpness=0.9):
        if not enabled:
            return
        reason = incompatibility(p, ps.samples)
        if reason:
            p.extra_generation_params["Steppe"] = "Skipped - " + reason
            gr.Warning("Steppe skipped: " + reason)
            return

        try:
            ps.samples = self.upscale(p, ps.samples, p.seeds, scale, model, sharpness)
        except InterruptedError:
            return

    @staticmethod
    @torch.inference_mode()
    def upscale(p, samples, seeds, scale, model, sharpness):
        def check_stop():
            if shared.state.interrupted or shared.state.skipped or shared.state.stopping_generation:
                raise InterruptedError("Steppe interrupted")

        try:
            check_stop()
            passes = {"2x": 1, "4x": 2}[scale]
            device = memory.get_torch_device()
            dtype = torch.bfloat16 if memory.should_use_bf16(device) else torch.float32
            upscaler = load_upscaler_file(model_path(model), dtype)
            if upscaler.info.latent_space != "flux_2":
                raise ValueError("Selected weights do not use the FLUX.2 latent format")
            patcher = ModelPatcher(upscaler.model, device, torch.device("cpu"))
            outputs = []
            previous_status = shared.state.textinfo
            try:
                # Upstream's activation estimate, for the largest pass of this request.
                grid_scale = upscaler.latent_space.patch_size * (2 ** passes)
                tokens = samples.shape[-2] * samples.shape[-1] * grid_scale ** 2
                config = upscaler.model.config
                activation_bytes = (torch.finfo(dtype).bits // 8) * (
                    10 * tokens * config.width + 6 * max(tokens, 65536) * config.mlp_hidden
                )
                memory.load_models_gpu([patcher], memory_required=activation_bytes, force_full_load=True)
                for index, sample in enumerate(samples):
                    result = sample.unsqueeze(0)
                    for n in range(passes):
                        check_stop()
                        shared.state.textinfo = f"Steppe {scale}, pass {n + 1}/{passes}"
                        settings = UpscaleSettings(seed=int(seeds[index]) + n, hff=float(sharpness))
                        result = upscaler.upscale(result, settings, device=device, on_pass=check_stop)
                    outputs.append(result)
            finally:
                shared.state.textinfo = previous_status
                memory.unload_model(patcher)
            result = torch.cat(outputs).to(samples)
        except InterruptedError:
            raise
        except Exception as error:
            p.extra_generation_params["Steppe"] = "Failed"
            gr.Warning(f"Steppe failed: {error}")
            raise
        p.extra_generation_params.update({"Steppe": scale, "Steppe model": model,
                                         "Steppe fine detail": sharpness,
                                         "Steppe output": f"{result.shape[-1]*16}x{result.shape[-2]*16}"})
        return result

    def postprocess(self, p, processed, *args):
        detail = getattr(p, "_steppe_flux3_detail", None)
        if detail:
            processed.infotexts = [f"{info}, {detail}" for info in processed.infotexts]
            processed.info = processed.infotexts[0] if processed.infotexts else processed.info


def flux3_callback(p):
    for script in getattr(getattr(p, "scripts", None), "alwayson_scripts", ()):
        if not isinstance(script, SteppeFlux2):
            continue
        args = p.script_args[script.args_from:script.args_to]
        if not args or not args[0]:
            return None
        if any(getattr(p, name, None) is not None for name in ("mask", "image_mask")):
            gr.Warning("Steppe skipped for masked FLUX.3 generation.")
            return None
        def run(packed, seed):
            from flux3_forge.runtime import Cancelled
            try:
                result = script.upscale(p, packed, [seed], *args[1:])
            except InterruptedError as error:
                raise Cancelled() from error
            p._steppe_flux3_detail = ", ".join(
                f"{key}: {p.extra_generation_params[key]}"
                for key in ("Steppe", "Steppe model", "Steppe fine detail", "Steppe output")
            )
            return result
        return run
    return None


def attach_flux3(*_):
    if "flux3_forge.bridge" in sys.modules:
        import steppe_flux3_bridge
        steppe_flux3_bridge.install(flux3_callback)


def detach_flux3():
    bridge = sys.modules.get("steppe_flux3_bridge")
    if bridge is not None:
        bridge.uninstall()


def save_flux3_metadata(params):
    detail = getattr(params.p, "_steppe_flux3_detail", None)
    if detail and params.pnginfo.get("parameters") and detail not in params.pnginfo["parameters"]:
        params.pnginfo["parameters"] += ", " + detail


script_callbacks.on_app_started(attach_flux3)
script_callbacks.on_script_unloaded(detach_flux3)
script_callbacks.on_before_image_saved(save_flux3_metadata)
