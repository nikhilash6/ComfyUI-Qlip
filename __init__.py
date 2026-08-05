from .nodes import QlipEnginesLoader, QlipLoraStack, QlipLoraSwitch
from .nodes import QlipTimerStart, QlipTimerStop, QlipTimerReport
from .nodes import QlipCache, QlipCacheReport
from .nodes import QlipAutoSparse
from .nodes import QlipProgressive
from .nodes import QlipCompile, QlipQuantConfig

NODE_CLASS_MAPPINGS = {
    "QlipEnginesLoader": QlipEnginesLoader,
    "QlipLoraStack": QlipLoraStack,
    "QlipLoraSwitch": QlipLoraSwitch,
    "QlipTimerStart": QlipTimerStart,
    "QlipTimerStop": QlipTimerStop,
    "QlipTimerReport": QlipTimerReport,
    "QlipCache": QlipCache,
    "QlipCacheReport": QlipCacheReport,
    "QlipAutoSparse": QlipAutoSparse,
    "QlipProgressive": QlipProgressive,
    "QlipCompile": QlipCompile,
    "QlipQuantConfig": QlipQuantConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QlipEnginesLoader": "Qlip Engines Loader",
    "QlipLoraStack": "Qlip LoRA Stack",
    "QlipLoraSwitch": "Qlip LoRA Switch",
    "QlipTimerStart": "Qlip Timer Start",
    "QlipTimerStop": "Qlip Timer Stop",
    "QlipTimerReport": "Qlip Timer Report",
    "QlipCache": "Qlip Cache",
    "QlipCacheReport": "Qlip Cache Report",
    "QlipAutoSparse": "Qlip Auto Sparse",
    "QlipProgressive": "Qlip Progressive",
    "QlipCompile": "Qlip Compile",
    "QlipQuantConfig": "Qlip Quant Config",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
