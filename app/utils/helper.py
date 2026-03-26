import random
import datetime
import secrets

def generate_otp():
    return f"{secrets.randbelow(10000):04d}"




# def generate_otp():
#     return str(random.randint(1000,9999))


def get_otp_expire_time(minutes=5):
    return datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=minutes)

