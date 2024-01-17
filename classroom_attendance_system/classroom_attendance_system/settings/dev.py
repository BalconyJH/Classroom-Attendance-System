from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-in@9&a=k7px6rebo@5%@7iph2s^2%^tqiqxk@$t9r125%8h0p-"

# SECURITY WARNING: define the correct hosts in production!
ALLOWED_HOSTS = ["*"]

EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
#
# # Sentry
# import sentry_sdk
#
# # SENTRY_DSN = "https://350b6b5c161d4d8c43f62c928a8ff5bf@o4505203476660224.ingest.sentry.io/4506584854364160"
#
# sentry_sdk.init(
#     dsn=config.sentry_dsn,
#     # Set traces_sample_rate to 1.0 to capture 100%
#     # of transactions for performance monitoring.
#     traces_sample_rate=1.0,
#     # Set profiles_sample_rate to 1.0 to profile 100%
#     # of sampled transactions.
#     # We recommend adjusting this value in production.
#     profiles_sample_rate=1.0,
#     enable_tracing=True,
# )
# print(config.sentry_dsn)

try:
    from .local import *
except ImportError:
    pass
