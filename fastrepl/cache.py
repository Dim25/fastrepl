class LLMCache:
    llm_cache: bool = True

    @staticmethod
    def enabled() -> bool:
        return LLMCache.llm_cache

    @staticmethod
    def enable():
        LLMCache.llm_cache = True

    @staticmethod
    def disable():
        LLMCache.llm_cache = False
