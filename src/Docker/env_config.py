import os


class Configuration:
    # def __init__(self) -> None:
        # required_envs = {
        #     'DB_PASSWORD',
        #     'EMAIL_PASSWORD'
        # }
        # missing_envs = required_envs - set(os.environ.keys())
        # if missing_envs:
        #     raise UserWarning(f'Missing required environment variables: {", ".join(missing_envs)}')

    @property
    def db_host(self):
        return os.getenv('DB_HOST', 'djangodb')

    @property
    def db_port(self):
        return os.getenv('DB_PORT', 3306)

    @property
    def db_user(self):
        return os.getenv('DB_USER', 'root')

    @property
    def db_password(self):
        return os.getenv('DB_PASSWORD', 'CK3/L}xs>=(Z(JEZ')

    @property
    def db_name(self):
        return os.getenv('DB_NAME', 'core')

    @property
    def email_host(self):
        return os.getenv('EMAIL_HOST', 'offconon.pro')

    @property
    def email_port(self):
        return os.getenv('EMAIL_PORT', 587)

    @property
    def email_user(self):
        return os.getenv('EMAIL_USER', 'no-reply@offconon.com')

    @property
    def email_password(self):
        return os.getenv('EMAIL_PASSWORD', 'ippN43G2qG')

    @property
    def email_page_domain(self):
        return os.getenv('EMAIL_PAGE_DOMAIN', 'https://api.offconon.com/')

    @property
    def secret_key(self):
        return os.getenv('SECRET_KEY',  'django-insecure-16u=o+q*=q-9s52ra4vsa0ftbjujae^cfuhfe)s%tzuc@k^_ce')
