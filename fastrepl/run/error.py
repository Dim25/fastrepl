import openai.error

from fastrepl.utils import pprint


def check_retryable_and_log(exp: Exception) -> bool:
    if isinstance(exp, openai.error.ServiceUnavailableError):
        pprint("[red]Service Unavailable[/red]")
    elif isinstance(exp, openai.error.APIError):
        pprint("[red]API Error[/red]")
    elif isinstance(exp, openai.error.RateLimitError):
        pprint("[red]Rate Limit Error[/red]")
    elif isinstance(exp, openai.error.APIConnectionError):
        pprint("[red]API Connection Error[/red]")
    elif isinstance(exp, openai.error.Timeout):
        pprint("[red]Timeout[/red]")
    else:
        return False
    return True


class RetryableException(Exception):
    def __init__(self):
        super(Exception, self).__init__("retryable exception from fastrepl")
