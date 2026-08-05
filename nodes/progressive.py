"""Qlip Progressive — low-resolution early steps via a model-call hook.

Thin ComfyUI router: ALL logic (interpolation rules, layout adapters for
plain/nested/flat-packed latents, the auto ladder with sigma backbone +
latent-size floor + x̂0-stability accelerator) lives in the licensed qlip
package — ``qlip.inference.progressive_core.ProgressiveEngine``. This node
only wires the engine into ComfyUI's official
``set_model_unet_function_wrapper`` (chaining with any wrapper set earlier,
e.g. QlipCache) and parses the timestep.

Sampling through the engine's low-resolution path runs under the standard
qlip paid-session gate, inherited from the loaded Qlip engines when
present.
"""

from .engine_loader import _validate_diffusion_model_input


class QlipProgressive:
    """Early denoising steps at low latent resolution — as a model hook."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Any DiT; keep your own "
                          "sampler — this is a model hook, not a sampler."}),
                "enable": ("BOOLEAN", {"default": True}),
                "low_scale": ("FLOAT", {"default": 0.5, "min": 0.25,
                              "max": 0.9, "step": 0.05, "tooltip":
                              "sigma mode ONLY. Latent side scale for the "
                              "low-res phase. auto mode ignores this and "
                              "always starts at 0.25, climbing by itself."}),
                "switch_at": ("FLOAT", {"default": 0.5, "min": 0.1,
                              "max": 0.9, "step": 0.05, "tooltip":
                              "sigma mode ONLY: fraction of the sigma "
                              "range spent at low resolution. auto mode "
                              "has no knobs (fixed sigma backbone + "
                              "latent-size floor + x̂0-stability "
                              "accelerator)."}),
            },
            "optional": {
                "switch_mode": (["auto", "sigma"], {"default": "auto",
                                "tooltip": "auto = adaptive ladder, no "
                                "knobs. sigma = fixed single switch at "
                                "switch_at with low_scale."}),
                "stab_threshold": ("FLOAT", {"default": 0.08, "min": 0.01,
                                   "max": 0.5, "step": 0.01, "tooltip":
                                   "Reserved; auto mode ignores this."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "qlip"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def apply(self, model, enable=True, low_scale=0.5, switch_at=0.5,
              switch_mode="auto", stab_threshold=0.08):
        _validate_diffusion_model_input(model, "QlipProgressive")
        patched = model.clone()
        if not enable:
            return (patched,)

        try:
            from qlip.inference.progressive_core import ProgressiveEngine
        except ImportError as exc:
            raise RuntimeError(
                "QlipProgressive requires the qlip package "
                f"(qlip.inference.progressive_core): {exc}") from exc

        engine = ProgressiveEngine(
            switch_mode=switch_mode, low_scale=low_scale,
            switch_at=switch_at, stab_threshold=stab_threshold)
        prev_wrapper = patched.model_options.get("model_function_wrapper")

        def unet_wrapper(apply_model, args):
            if prev_wrapper is not None:
                base_apply = apply_model

                def apply_model(x, t, **c):     # noqa: F811 — chained inner
                    return prev_wrapper(base_apply, {
                        "input": x, "timestep": t, "c": c,
                        "cond_or_uncond": args.get("cond_or_uncond", [0])})
            x = args["input"]
            timestep = args["timestep"]
            c = args["c"]
            try:
                sig = float(timestep.reshape(-1)[0])
            except Exception:
                return apply_model(x, timestep, **c)
            return engine.step(
                x, timestep, sig, c, apply_model,
                lane_key=tuple(args.get("cond_or_uncond", [0])))

        patched.set_model_unet_function_wrapper(unet_wrapper)
        print(f"[QlipProgressive] enabled: mode={switch_mode} (licensed "
              f"qlip engine) — early model calls run on a downscaled "
              f"latent; nested multimodal latents supported; your sampler "
              f"is untouched.")
        return (patched,)
