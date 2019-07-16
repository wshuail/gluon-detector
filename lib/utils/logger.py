import logging

def build_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = '%(asctime)s;%(message)s'
    formater = logging.Formatter(fmt=fmt)

    fh = logging.FileHandler(filename=log_path)
    fh.setFormatter(formater)
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setFormatter(formater)
    # logger.addHandler(sh)

