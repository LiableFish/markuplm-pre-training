from typing import Any, Dict, List
from transformers import DataCollatorForLanguageModeling


class DataCollatorForMarkupLMLanguageModelling(DataCollatorForLanguageModeling):
    def tf_call(self, examples: List[List[int] | Any | Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support TensorFlow")
    
    def torch_call(self, examples: List[List[int] | Any | Dict[str, Any]]) -> Dict[str, Any]:
        return super().torch_call(examples)
    
    def numpy_call(self, examples: List[List[int] | Any | Dict[str, Any]]) -> Dict[str, Any]:
        return super().numpy_call(examples)