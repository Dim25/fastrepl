import openai.error

from fastrepl.utils import pprint


# TODO: This logging should be added to some global context and rendered in Live display
def check_retryable_and_log(exp: Exception) -> bool:
    if isinstance(exp, openai.error.ServiceUnavailableError):
        pprint("[bright_red]Service Unavailable[/bright_red]")
    elif isinstance(exp, openai.error.APIError):
        pprint("[bright_red]API Error[/bright_red]")
    elif isinstance(exp, openai.error.RateLimitError):
        pprint("[bright_red]Rate Limit Error[/bright_red]")
    elif isinstance(exp, openai.error.APIConnectionError):
        pprint("[bright_red]API Connection Error[/bright_red]")
    elif isinstance(exp, openai.error.Timeout):
        pprint("[bright_red]Timeout[/bright_red]")
    else:
        return False
    return True


class RetryableException(Exception):
    def __init__(self):
        super(Exception, self).__init__("retryable exception from fastrepl")
